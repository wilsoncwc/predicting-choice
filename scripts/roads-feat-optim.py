#!/usr/bin/env python
# coding: utf-8

# In[40]:


import math
import copy
import os.path as osp
from itertools import chain

import momepy
import fiona
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList

from torch_geometric.data import Data, Batch, InMemoryDataset, download_url, extract_zip
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GCNConv, GAE, VGAE, global_sort_pool
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T

from torch.utils.tensorboard import SummaryWriter

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_root = '/homes/wwc4618/predicting-choice'
dataset_root = f'{project_root}/datasets'

full_dataset_label = 'No Bounds'
training_places = ['London']

# Fields to ignore
meridian_fields = ['meridian_id', 'meridian_gid', 'meridian_code',
                   'meridian_osodr', 'meridian_number', 'meridian_road_name',
                   'meridian_indicator', 'meridian_class', 'meridian_class_scale']
census_geom_fields = ['wz11cd', 'lsoa11nm', 'msoa11nm',
                      'oa11cd', 'lsoa11cd', 'msoa11cd'] # Allowed: lad11cd, lad11nm
misc_fields = ['id']
ignore_fields = meridian_fields + census_geom_fields + misc_fields


unnorm_feature_fields = ['metres', 'choice2km', 'nodecount2km', 'integration2km',
                      'choice10km', 'nodecount10km','integration10km',
                      'choice100km','nodecount100km','integration100km']
rank_fields = ['choice2kmrank', 'choice10kmrank','integration10kmrank', 'integration2kmrank']
log_fields = ['choice2kmlog','choice10kmlog','choice100kmlog']
all_feature_fields = unnorm_feature_fields + rank_fields + log_fields

# dictionary caches - RELOADING THIS CELL DELETES GRAPH CACHES
loaded_gdfs = {}
loaded_graphs={}


# In[3]:


def process_graph(g, feature_fields=[]):
    """
    Takes SSx metrics from adjacent edges and averages them to form the node attribute
    """
    for node, d in g.nodes(data=True):
        # get attributes from adjacent edges
        new_data = {field:[] for field in feature_fields}
        for _, _, edge_data in g.edges(node, data=True):
            for field in feature_fields:
                new_data[field].append(edge_data[field])
        
        d.clear()
        # take the average of the collected attributes
        for field in feature_fields:
            d[field] = sum(new_data[field]) / len(new_data[field])
        
        # Encode node coordinate as feature
        d['x'], d['y'] = node
    
    for u, v, d in g.edges(data=True):
        d.clear()
        # Encode edge information for later indexing
        d['u'] = u
        d['v'] = v
    return g

def load_graph(place, feature_fields=[], reload=True, verbose=False):
  if verbose:
    print(f'Loading graph of {place}...')
  key = (place)
  if key in loaded_graphs and reload:
    g = loaded_graphs[key]
    if verbose:
        print('Loaded existing graph.')
    print(g)
  else:
    gdf = load_gdf(place, verbose=verbose)
    G = momepy.gdf_to_nx(gdf, approach='primal', multigraph=False)
    G = process_graph(G, feature_fields)
    
    if verbose:
      print(f'Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges')
    node_attrs = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))
    assert node_attrs == set(feature_fields + ['x', 'y'])
    node_attrs = feature_fields + ['x', 'y']
    edge_attrs = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
    
    if verbose:
        # List node attributes
        print(f'Node attributes: {node_attrs}')
        print(f'Edge attributes: {edge_attrs}')                            
    
    # If no node attributes, node degree will be added later
    # Edge attribute (angle) are not added
    if len(node_attrs) > 0:
        g = from_networkx(G, group_node_attrs=node_attrs)
    else:
        g = from_networkx(G)
    loaded_graphs[key] = g
  return g

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


feats = ['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km',
         'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km',
         'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank',
         'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog', 'x', 'y']
max_deg = 16
def process_dataset(full_dataset, split=0.8, hold_out_edge_ratio=0.2, batch_size=64,
                    include_feats=feats, add_deg_feats=False, verbose=False):
    """
        Set split < 0 to indicate that only the first graph should be used for training
    """
    dataset = copy.deepcopy(full_dataset)
    
    # Train-test split
    if split < 0:
        train_idx = [0]
        test_idx = list(range(1, len(full_dataset)))
        batch_size = 1
    else:
        idx = torch.randperm(len(dataset))
        split_idx = math.floor(split * len(dataset))
        train_idx = idx[:split_idx]
        test_idx = idx[split_idx:]
    
    # Remove non included features
    if include_feats != feats:
        feat_idx = torch.tensor([feats.index(feat) for feat in include_feats])
        for data in dataset:
            data.x = torch.index_select(data.x, 1, feat_idx) \
                     if len(feat_idx) > 0 else None
    
    # If no features are selected, inject degree profile as features
    # else, normalize the features
    neg_sampling_ratio = 1 / hold_out_edge_ratio
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.LocalDegreeProfile() if add_deg_feats else (lambda x: x),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0, num_test=0, disjoint_train_ratio=hold_out_edge_ratio,
                          neg_sampling_ratio=neg_sampling_ratio, is_undirected=True, split_labels=True)
    ])

    train_dataset = [transform(dataset[i])[0] for i in train_idx]
    test_dataset = [transform(dataset[i])[0] for i in test_idx]
    if verbose:
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')

    # Load graphs into dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if verbose:
        for step, data in enumerate(train_loader):
            print(f'Step {step + 1}:')
            print('=======')
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print(data)
            print()

    return train_dataset, test_dataset, train_loader, test_loader

def train(model, optimizer, loader, variational):
    model.train()
    loss_tot = 0
    loss_recon = 0
    loss_kl = 0
    for data in loader:
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        
        loss = model.recon_loss(z, data.edge_index, data.neg_edge_label_index)
        loss_recon += loss.item()
        
        if variational:
            kl_loss = (1 / data.num_nodes) * model.kl_loss()
            loss_kl += kl_loss.item()
            loss += kl_loss
        
        loss_tot += loss.item()
        loss.backward()
        optimizer.step()
    loss_tot /= len(loader.dataset)
    loss_recon /= len(loader.dataset)
    loss_kl /= len(loader.dataset)
    return loss_tot, loss_recon, loss_kl


@torch.no_grad()
def test(model, loader):
    model.eval()
    auc_tot = 0
    ap_tot = 0
    for data in loader:
        z = model.encode(data.x, data.edge_index)
        auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        auc_tot += auc
        ap_tot += ap
    return auc_tot / len(loader), ap_tot / len(loader)


# In[46]:


metric_dict = {
    'total_loss': 'Total Loss',
    'recon_loss': 'Reconstruction Loss',
    'kl_loss': 'KL Divergence Loss',
    'train_auc': 'Transductive AUROC',
    'train_ap': 'Transductive Average Precision',
    'test_auc': 'Inductive AUROC',
    'test_ap': 'Inductive Average Precision'
}


def init_model(linear, variational, *model_args):
    if not variational and not linear:
        model = GAE(GCNEncoder(*model_args))
    elif not variational and linear:
        model = GAE(LinearEncoder(*model_args))
    elif variational and not linear:
        model = VGAE(VariationalGCNEncoder(*model_args))
    elif variational and linear:
        model = VGAE(VariationalLinearEncoder(*model_args))
        
    model = model.to(device)
    return model

def run(
    dataset,
    num_iter=5,
    save_best_model=False,
    save_best_model_metric='test_auc',
    output_tb=False,
    tag_tb='',
    batch_size=32,
    include_feats=feats,
    add_deg_feats=True,
    epochs = 1000,
    print_every = 10,
    out_channels = 5,
    lr = 0.001,
    linear = False,
    variational = False
):
    
    train_dataset, test_dataset, train_loader, test_loader = process_dataset(dataset,
                                                                         batch_size=batch_size,
                                                                         include_feats=include_feats,
                                                                         add_deg_feats=add_deg_feats)
    in_channels = train_dataset[0].num_node_features
    
    mod = 'V' if variational else ''
    mod += 'L' if linear else ''
    tag_tb = f'{"deg+" if add_deg_feats else ""}{len(include_feats)}_{tag_tb}'
    run_str = f'G{mod}AE_{out_channels}d_{epochs}epochs_{lr}lr_{tag_tb}feats'
    
    results = []
    models = []
    for i in range(1, num_iter + 1):
        print(f'Running iteration {i} of expt {run_str}')
        model = init_model(linear, variational, in_channels, out_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        result_dict = {metric: [] for metric in metric_dict}
        for epoch in range(1, epochs + 1):
            total_loss, recon_loss, kl_loss = train(model, optimizer, train_loader, variational)
            train_auc, train_ap = test(model, train_dataset)
            test_auc, test_ap = test(model, test_dataset)

            result_dict['total_loss'].append(total_loss)
            result_dict['recon_loss'].append(recon_loss)
            result_dict['kl_loss'].append(kl_loss)
            result_dict['train_auc'].append(train_auc)
            result_dict['train_ap'].append(train_ap)
            result_dict['test_auc'].append(test_auc)
            result_dict['test_ap'].append(test_ap)
            if epoch % print_every == 0:
                print(f'Epoch {epoch:03d}: Train AUC: {train_auc:.4f}, Train AP: {train_ap:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')
        results.append(result_dict)
        models.append(model)

    if output_tb:
        path = f'{project_root}/runs/{run_str}'
        print(f'Writing to {path}')
        writer = SummaryWriter(path)

        for metric in metric_dict:
            mean_result = np.mean([res[metric] for res in results], axis=0)
            for i, result in enumerate(mean_result):
                writer.add_scalar(metric_dict[metric], result, i)

    if save_best_model:
        results_best_metric = [res[save_best_model_metric][-1] for res in results]
        best_iteration = np.argmax(results_best_metric)
        
        path = f'{project_root}/models/{run_str}.pt'
        print(f'Saving model to {path}')
        torch.save(models[best_iteration].state_dict(), path)
    
    return models, results

def main():
    seed_everything(42)
    dataset = torch.load(f'{dataset_root}/ssx_dataset.pt')
    
    experiments = {
        'variational': [False],
        'lr': [0.01],
        'out_channels': [10],
        'add_deg_feats': [False, True],
        'included_feats': [rank_fields, rank_fields + ['x', 'y']]
    }

    for var in experiments['variational']:
        for lr in experiments['lr']:
            for oc in experiments['out_channels']:
                for adf in experiments['add_deg_feats']:
                    for i_f in experiments['included_feats']:
                        if not adf and len(i_f)==0:
                            continue
                        run(dataset,
                            variational=var, 
                            lr=lr, 
                            out_channels=oc, 
                            add_deg_feats=adf,
                            include_feats=i_f,
                            output_tb=True,
                            tag_tb='neg',
                            save_best_model=True)

if __name__ == '__main__':
    main()
