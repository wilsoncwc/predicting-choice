#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import time
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
from torch_geometric.nn import MLP, GCNConv, global_sort_pool
from torch_geometric.nn import GAE, VGAE
from torch_geometric.transforms import RandomLinkSplit, OneHotDegree
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
from torch_geometric.utils.convert import from_networkx

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
dataset_root = '/vol/bitbucket/wwc4618/datasets'


# In[2]:


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

# Post-processing features
feats = ['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km',
         'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km',
         'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank',
         'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog', 'x', 'y']

# dictionary caches - RELOADING THIS CELL DELETES GRAPH CACHES
loaded_gdfs = {}
loaded_graphs={}

def load_gdf(place=training_places[0], verbose=False): #(W, S, E, N)
    """Geodataframe (gdf) loader with caching"""
    if place in included_places:
        # Retrieve matching rows corresponding to the Local Authority
        print(f'Loading {place} from SSx')
        gdf = full_gdf.query(f'lad11nm == "{place}"').copy()
    elif place == full_dataset_label:
        # Read full dataset without boundaries
        gdf = full_gdf.copy()
    else:
        print(f'Loading {place} from OSMnx')
        # Load gdf from osmnx (for testing only, graph will lack target attr)
        # Actually uses the Nominatim API:
        # https://nominatim.org/release-docs/latest/api/Overview/
        g = ox.graph.graph_from_place(place, buffer_dist=osmnx_buffer)
        g = ox.projection.project_graph(g)
        gdf = ox.utils_graph.graph_to_gdfs(g, nodes=False)
        gdf = gdf.rename(columns={'length': 'metres'})
        return gdf

    if verbose:
        print(f'{gdf.size} geometries retrieved from {place}')

    loaded_gdfs[place] = gdf
    return gdf


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
    g.place = place
  return g

def construct_ssx_dataset():
    loaded_graphs ={}
    dataset = [ load_graph(place, all_feature_fields, verbose=True)                 for place in included_places ]
    torch.save(dataset, 'datasets/ssx_dataset.pt')
    return dataset


# # GVAE

# In[4]:


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
    
class MyGAE(GAE):
    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_fscore_support
            

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        
        # Calculate best threshold for edge labelling, based on geometric mean of sensitivity/specificity 
        fpr, tpr, thresholds = roc_curve(y, pred)
        gmeans = np.sqrt(tpr * (1-fpr))
        best_threshold = thresholds[np.argmax(gmeans)]
        thresholded_pred = [0 if logit < best_threshold else 1 for logit in pred]
        
        return roc_auc_score(y, pred), average_precision_score(y, pred), precision_recall_fscore_support(y, thresholded_pred, average='binary')
    


# In[5]:


from torch_geometric.loader import DataLoader    
import copy
import torch_geometric.transforms as T

max_deg = 16
def process_dataset(full_dataset, split=0.8, split_labels=True, hold_out_edge_ratio=0.2,
                    neg_sampling_ratio=-1, batch_size=32, include_feats=feats,
                    add_deg_feats=False, deg_after_split=False, verbose=False):
    """
        Set split < 0 to indicate that only the first graph should be used for training
    """
    dataset = copy.deepcopy(full_dataset)
    
    # Train-test split
    if split < 0:
        train_idx = [0]
        test_idx = list(range(1, len(dataset)))
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
            data.num_nodes = data.x.size(0)
            data.x = torch.index_select(data.x, 1, feat_idx) \
                     if len(feat_idx) > 0 else None
    
    # If no features are selected, inject degree profile as features
    # else, normalize the features
    neg_sampling_ratio = neg_sampling_ratio if neg_sampling_ratio >= 0 else 1 / hold_out_edge_ratio
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.LocalDegreeProfile() if add_deg_feats and not deg_after_split else (lambda x: x),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0, num_test=0, disjoint_train_ratio=hold_out_edge_ratio,
                          neg_sampling_ratio=neg_sampling_ratio, split_labels=split_labels, is_undirected=True)
    ])

    train_dataset = [transform(dataset[i])[0] for i in train_idx]
    test_dataset = [transform(dataset[i])[0] for i in test_idx]
    if deg_after_split and add_deg_feats:
        deg_xform = T.Compose([T.LocalDegreeProfile(), T.ToDevice(device)])
        train_dataset = [deg_xform(data) for data in train_dataset]
        test_dataset = [deg_xform(data) for data in test_dataset]
    
    if verbose:
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')

    # Load graphs into dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) \
        if len(train_dataset) > 0 else []
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) \
        if len(test_dataset) > 0 else []
    if verbose:
        for step, data in enumerate(train_loader):
            print(f'Step {step + 1}:')
            print('=======')
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print(data)
            print()

    return train_dataset, test_dataset, train_loader, test_loader


# In[6]:


from torch.utils.tensorboard import SummaryWriter

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


# In[7]:


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
    data_process_args={},
    num_iter=5,
    save_best_model=False,
    save_best_model_metric='test_ap',
    output_tb=False,
    tag_tb='',
    epochs = 1000,
    print_every = 10,
    out_channels = 5,
    lr = 0.001,
    linear = False,
    variational = False
):
    """
        data_process_args: dict of optional keywords as follows:
            split, hold_out_edge_ratio, batch_size, include_feats, add_deg_feats
    """
    if type(dataset) is tuple:
        # Assume already processed
        train_dataset, test_dataset, train_loader, test_loader = dataset
    else:
        train_dataset, test_dataset, train_loader, test_loader = process_dataset(dataset, **data_process_args)
    in_channels = train_dataset[0].num_node_features
    
    mod = 'V' if variational else ''
    mod += 'L' if linear else ''
    add_deg_feats = data_process_args['add_deg_feats']
    include_feats = data_process_args['include_feats']
    tag_tb = f'{"deg+" if add_deg_feats else ""}{len(include_feats)}_{tag_tb}'
    run_str = f'G{mod}AE_{out_channels}d_{epochs}epochs_{lr}lr_{tag_tb}feats'
    
    results = []
    models = []
    start_time = time.time()
    for i in range(1, num_iter + 1):
        print(f'Running iteration {i} of expt {run_str}')
        model = init_model(linear, variational, in_channels, out_channels)
        model.data_process_args = data_process_args
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
        result_dict = {metric: [] for metric in metric_dict}
        for epoch in range(1, epochs + 1):
            total_loss, recon_loss, kl_loss = train(model, optimizer, train_loader, variational)
            train_auc, train_ap = test(model, train_dataset)
            test_auc, test_ap = test(model, test_dataset)
            scheduler.step(total_loss)
            
            result_dict['total_loss'].append(total_loss)
            result_dict['recon_loss'].append(recon_loss)
            result_dict['kl_loss'].append(kl_loss)
            result_dict['train_auc'].append(train_auc)
            result_dict['train_ap'].append(train_ap)
            result_dict['test_auc'].append(test_auc)
            result_dict['test_ap'].append(test_ap)
            if epoch % print_every == 0:
                sec_per_epoch = (time.time() - start_time) / print_every
                start_time = time.time()
                print(f'Epoch {epoch:03d} ({sec_per_epoch:.2f}s/epoch): '
                f'Train AUC: {train_auc:.4f}, Train AP: {train_ap:.4f},'
                f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')
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


# # Train single

# In[13]:


from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score

GLOBAL_THRESHOLD = 0.6

def test_model(processed_dataset, model):
    test_result_dict = {}
    for test_data in processed_dataset:
        result_dict = {}
        # test_data = process_dataset([data], verbose=False, split_labels=False, 
        #                             **model.data_process_args)[0][0]
        z = model.encode(test_data.x, test_data.edge_index)
        result_dict['z'] = z.detach().cpu()
        out_enc_edges = model.decode(z, test_data.edge_index).view(-1)
        out_test_edges = model.decode(z, test_data.edge_label_index).view(-1)

        out = torch.cat([out_enc_edges, out_test_edges]).detach().cpu()
        y_label = torch.ones(test_data.edge_index.size(1))
        edge_label = torch.cat([y_label, test_data.edge_label.detach().cpu()])

        # Pre-threshold metrics
        result_dict['auc'] = roc_auc_score(edge_label, out.detach())
        result_dict['ap'] = average_precision_score(edge_label, out.detach())
        
        # Post-threshold metrics
        #Calculate best threshold for edge labelling that gives the highest F1
        thresholded_pred = (out > GLOBAL_THRESHOLD).float()
        result_dict['precision'] = precision_score(edge_label, thresholded_pred)
        result_dict['recall'] = recall_score(edge_label, thresholded_pred)
        result_dict['f1'] = f1_score(edge_label, thresholded_pred)
        # precision, recall, thresholds = roc_curve(edge_label, out.detach())
        # gmeans = np.sqrt(tpr * (1-fpr))
        # best_threshold = thresholds[np.argmax(gmeans)]
        # thresholded_pred = (out > best_threshold).float
        # result_dict['precision'], result_dict['recall'], result_dict['thresholds'] = precision_recall_curve(edge_label, thresholded_pred)
        
        test_result_dict[test_data.place] = result_dict
    return test_result_dict

def train_single(places, full_dataset, run_args={}, save_path=''):
    single_data = {}
    data_process_args = {
        'split': 0, # indicates only the first graph should be used for training
        'batch_size': 32,
        'add_deg_feats': run_args.pop('add_deg_feats', False),
        'deg_after_split': run_args.pop('deg_after_split', False),
        'include_feats': run_args.pop('include_feats', rank_fields)
    }
    _, test_dataset, _, test_loader = process_dataset(full_dataset, verbose=False, **data_process_args)
    post_run_test_dataset = process_dataset(full_dataset, verbose=False, split_labels=False, **data_process_args)[1]
    for place in places:
        print(f'Training model on SSx data from {place}...')
        place_graph = next(data for data in test_dataset if data.place == place)
        train_dataset = [place_graph] # train on a single graph only
        train_loader = DataLoader(train_dataset, batch_size=1)
        dataset = train_dataset, test_dataset, train_loader, test_loader
        
        models, results = run(dataset, data_process_args, **run_args)
        
        print('Model training ended. Computing metrics...')
        data = {}
        # Append final metrics of each run
        for result in results:
            for key in result:
                if key in data:
                    data[key].append(result[key][-1])
                else:
                    data[key] = [result[key][-1]]
                
        # Test models on every city and compute average over models
        test_dict = {}
        for model in models:
            metrics = test_model(post_run_test_dataset, model)
            for result_place in metrics:
                result_metrics = metrics[result_place]
                if result_place in test_dict:
                    for metric in result_metrics:
                        test_dict[result_place][metric].append(result_metrics[metric])
                else:
                    test_dict[result_place] = {metric_name: [metric]
                                        for metric_name, metric in result_metrics.items()}
        for result_place in test_dict:
            result_metrics = test_dict[result_place]
            for metric in result_metrics:
                if metric == 'z':
                    continue
                result_metrics[metric] = sum(result_metrics[metric]) / len(result_metrics[metric])
        
        # Save transductive (same place) and inductive-average (over all places) metrics
        metrics = ['auc', 'ap', 'precision', 'recall', 'f1']
        avg_dict = { metric: 0 for metric in metrics }
        for place2 in test_dict:
            for metric in metrics:
                avg_dict[metric] += test_dict[place2][metric]
        for metric in metrics:
            avg_dict[metric] /= len(test_dict)
        data['transductive'] = test_dict[place]
        data['inductive-avg'] = avg_dict

        # data['tests'] = test_dict (Too big to be saved)
        single_data[place] = data
        print(single_data.keys())
        if save_path:
            torch.save(single_data, save_path)

    return single_data


def main():
    seed_everything(42)
    dataset = torch.load(f'{dataset_root}/ssx_dataset.pt')
    places = [data.place for data in dataset]
    half = len(places) // 2
    places = places[half:]
    
    run_hyperparams = {
        'num_iter': 1,
        'lr': 0.02,
        'epochs': 200,
        'out_channels': 10,
        'add_deg_feats': True,
        'include_feats': []
    }
    train_single(places, dataset, run_args=run_hyperparams,
                     save_path=f'{dataset_root}/link_pred_metrics_2_iter2.pt')

if __name__ == '__main__':
    main()

