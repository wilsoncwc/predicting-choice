#!/usr/bin/env python
# coding: utf-8

# In[40]:


import math
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
    
seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_root = '/homes/wwc4618/predicting-choice'
dataset_root = f'{project_root}/datasets'


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
  return g

def construct_ssx_dataset():
    loaded_graphs ={}
    dataset = [ load_graph(place, all_feature_fields, verbose=True)                 for place in included_places ]
    torch.save(dataset, 'datasets/ssx_dataset.pt')
    return dataset


# In[4]:


from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T

from sklearn.metrics import roc_curve

def visualize_preds(place, model, add_deg_feats=False):
    data = load_graph(place, all_feature_fields)

    # Randomly sample negative links and run predictions
    hold_out_edge_ratio = 0.2 # Remove some edges for testing
    neg_sampling_ratio = 1 # Negative-sample original edge count
    transforms = T.Compose([
        T.NormalizeFeatures(),
        T.LocalDegreeProfile() if add_deg_feats else (lambda x: x),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0, num_test=0, disjoint_train_ratio=hold_out_edge_ratio,
                          neg_sampling_ratio=neg_sampling_ratio, is_undirected=True)
    ])
    vis_data, _, _ = transforms(data.clone())
    vis_data = vis_data.to(device)
    z = model.encode(vis_data.x, vis_data.edge_index)
    out_enc_edges = model.decode(z, vis_data.edge_index).view(-1)
    out_test_edges = model.decode(z, vis_data.edge_label_index).view(-1)
    
    out = torch.cat([out_enc_edges, out_test_edges]).cpu()
    y_label = torch.ones(vis_data.edge_index.size(1))
    edge_label = torch.cat([y_label, vis_data.edge_label.cpu()])
    
    # Calculate best threshold for edge labelling, based on geometric mean of sensitivity/specificity 
    fpr, tpr, thresholds = roc_curve(edge_label, out.detach())
    gmeans = np.sqrt(tpr * (1-fpr))
    best_threshold = thresholds[np.argmax(gmeans)]
    print(f'Best threshold: {best_threshold}')
    
    cat_index = torch.cat([vis_data.edge_index, vis_data.edge_label_index], dim=-1)
    print('Finished prediction, rebuilding road network')
    
    res_dict = {} # For storing new edge attributes in nx
    pred_dict = {} # Map of coords to predicted values
    label_dict = {} # Map of coords to labels (sanity check)
    gdf = load_gdf(place)
    gdf.plot()
    plt.show()
    streets = momepy.gdf_to_nx(gdf, approach='primal', multigraph=False)
    float32_node_dict = {(torch.tensor(c[0], dtype=torch.float32).item(),
                          torch.tensor(c[1], dtype=torch.float32).item()): c for c in streets}
    count_dict = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    
    # Iterate over predicted edges (which includes real and fake edges)
    for i, logit in enumerate(out):
        # Get indices of both the nodes of the edge
        u_idx, v_idx = cat_index[:, i] 
        
        # Get their coordinates (the last two node attributes in pretransformed data)
        u_float32 = data.x[u_idx, -2].item(), data.x[u_idx, -1].item()
        v_float32 = data.x[v_idx, -2].item(), data.x[v_idx, -1].item()
        # Convert them into their full precision node coordinates
        u, v = float32_node_dict[u_float32], float32_node_dict[v_float32]
        key = (u, v)
        
        label = edge_label[i]
        if (u, v) in pred_dict:
            # Should NOT happen
            raise NotImplementedError
        elif (v, u) in pred_dict:
            logit = (pred_dict[(v, u)] + logit) / 2
        else:
            pred_dict[key] = logit
        pred = 0 if logit < best_threshold else 1
        
        if not (streets.has_edge(u, v) or streets.has_edge(v, u)):
            # Negative sampled edge
            assert label == 0
            res = 'tn' if pred == label else 'fp'
            
            # Add the false positive edges for visualization
            if res == 'fp':
                streets.add_edge(u, v, res=res)
        else:
            if label != 1:
                # Abort mission
                raise NotImplementedError(u, v, label)
            res = 'tp' if pred == label else 'fn'
            res_dict[(u, v)] = res

        count_dict[res] += 1
    print('Results')
    print(count_dict)

    # Set attributes on the original graph
    nx.set_edge_attributes(streets, res_dict, 'res')
    
    # lines = momepy.nx_to_gdf(gdf, approach='primal', multigraph=False)
    color_state_map = {'tp': 'green', 'fp': 'blue', 'fn': 'red'}
    colors = [color_state_map[edge[2]['res']] for edge in streets.edges(data=True)]
    nx.draw(streets, {n:[n[0], n[1]] for n in list(streets.nodes)}, node_size=0, edge_color=colors,
            edge_cmap='Set1')


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


from torch_geometric.loader import DataLoader    
import copy
import torch_geometric.transforms as T

feats = ['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km',
         'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km',
         'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank',
         'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog', 'x', 'y']
max_deg = 16
def process_dataset(full_dataset, split=0.8, hold_out_edge_ratio=0.2, batch_size=64,
                    include_feats=feats, add_deg_feats=False, verbose=False):
    dataset = copy.deepcopy(full_dataset)

    # Train-test split
    idx = torch.randperm(len(dataset))
    split_idx = math.floor(split * len(dataset))
    train_idx = idx[:split_idx]
    test_idx = idx[split_idx:]
    
    # Remove non included features
    if include_feats != feats:
        feat_idx = torch.tensor([feats.index(feat) for feat in include_feats])
        for data in dataset:
            data.x = torch.index_select(data.x, 1, feat_idx)                      if len(feat_idx) > 0 else None
    
    # If no features are selected, inject degree profile as features
    # else, normalize the features
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.LocalDegreeProfile() if add_deg_feats else (lambda x: x),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0, num_test=0, disjoint_train_ratio=hold_out_edge_ratio,
                          is_undirected=True, split_labels=True)
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

dataset = torch.load(f'{dataset_root}/ssx_dataset.pt')
train_dataset, test_dataset, train_loader, test_loader = process_dataset(dataset,
                                                                         batch_size=32,
                                                                         add_deg_feats=True,
                                                                         verbose=True)


# In[36]:


from torch_geometric.nn import GAE, VGAE
from torch.utils.tensorboard import SummaryWriter

def train(model, optimizer, loader, variational):
    model.train()
    loss_tot = 0
    loss_recon = 0
    loss_kl = 0
    for data in loader:
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        
        loss = model.recon_loss(z, data.edge_index)
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
    num_iter=5,
    save_best_model=False,
    save_best_model_metric='test_auc',
    output_tb=False,
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
    tag_tb = f'{"deg+" if add_deg_feats else ""}{len(include_feats)}'
    run_str = f'G{mod}AE_exp_{out_channels}d_{epochs}_epochs_{lr}_lr_{tag_tb}'
    
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

experiments = {
    'variational': [False, True],
    'lr': [0.01, 0.001],
    'out_channels': [2, 5, 10, 20],
    'add_deg_feats': [False, True],
    'included_feats': [[], ['x', 'y'], rank_fields, feats]
}

for var in experiments['variational']:
    for lr in experiments['lr']:
        for oc in experiments['out_channels']:
            for adf in experiments['add_deg_feats']:
                for i_f in experiments['included_feats']:
                    if not adf and len(i_f)==0:
                        continue
                    run(variational=var, 
                        lr=lr, 
                        out_channels=oc, 
                        add_deg_feats=adf,
                        include_feats=i_f,
                        output_tb=True,
                        save_best_model=True)


