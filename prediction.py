#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import time
import warnings
import networkx as nx
import numpy as np
import geopandas as gpd
import pandas as pd
import osmnx as ox
import momepy
import fiona
import scipy as sp
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, r2_score, mean_absolute_error
from imblearn.metrics import macro_averaged_mean_absolute_error
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from IPython.display import display
from numbers import Number
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader, NeighborLoader, GraphSAINTRandomWalkSampler, ShaDowKHopSampler

from models.init_model import init_gnn_model
from utils.utils import remove_item
from utils.seed import seed_everything
from utils.load_geodata import load_graph, load_gdf
from utils.constants import rank_fields, all_feature_fields, geom_feats, ignore_non_accident_field

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reg_loss_fns = {
    'mse': torch.nn.MSELoss(),
    'mae': torch.nn.L1Loss(),
    'huber': torch.nn.HuberLoss(),
    'poisson': torch.nn.PoissonNLLLoss()
}

def macro_mae(y_true, y_pred):
    # Round predictions to the nearest integer
    return macro_averaged_mean_absolute_error(y_true, torch.round(y_pred))

criteria_fns = {
    'MAE': mean_absolute_error,
    'Macro MAE': macro_mae,
    'R2': r2_score,
    'Accuracy': accuracy_score,
    'Macro Recall': balanced_accuracy_score,
    'F1': f1_score,
    'MCC': matthews_corrcoef
}
THRESHOLD = 0.5

def poissonLoss(xbeta, y):
    """Custom loss function for Poisson model."""
    loss=torch.mean(torch.exp(xbeta)-y*xbeta)
    return loss


def convert_accident_counts(tensor_data, categorize):
    if categorize == 'classification':
        labels = ['No Accidents', 'One Accident', 'Two Accidents', 'Few Accidents', 'Many Accidents']
    elif categorize == 'regression':
        labels = [0., 1., 2., 3., 4.]
        labels = [torch.tensor(label) for label in labels]
    elif categorize == 'multiclass':
        labels = [
            torch.tensor([0., 0., 0., 0.]),
            torch.tensor([1., 0., 0., 0.]),
            torch.tensor([1., 1., 0., 0.]),
            torch.tensor([1., 1., 1., 0.]),
            torch.tensor([1., 1., 1., 1.])
        ]
    else:
        return tensor_data
    
    print(f'Categorizing accident_count')
    bins = [0, 1, 2, 3, 5, np.inf]
    transformer = FunctionTransformer(
        pd.cut, kw_args={'bins': bins, 'labels': False, 'retbins': False, 'right': False}
    )
    transformed_data = transformer.fit_transform(tensor_data)
    
    return [labels[label_idx] for label_idx in transformed_data]

def load_data(
    place,
    target_field,
    categorize=None,
    include_feats=None,
    cat_feats=[],
    add_deg_feats=False,
    clean=True,
    dist=50,
    agg='sum',
    split=0.8,
    batch_size=64,
    split_approach=None,
    num_parts=None, # For split_approach='cluster'
    num_neighbors=5, # For split_approach 'saint', 'neighbor'
    graph_length=2, # For split_approach 'saint', 'neighbor' (size of subgraph in hops)
    return_nx=False,
    verbose=False
):
    """
    Loads and prepares the graph of roads from `place` for node classification.
    
    Args:
        place (str): Name of Local Authority, or 'No Bounds' for full graph,
        target_field (str): Name of target attribute,
        categorize (str, optional): Classification strategy, takes one of ['classification', ],
        include_feats (list, optional),
        cat_feats (list, optional),
        add_deg_feats (bool, optional),
        clean (bool, optional),
        dist (int, optional),
        agg (str, optional),
        split (float, optional),
        batch_size (int, optional),
        split_approach (str, optional),
        num_parts (int, optional) # For split_approach='cluster'
        num_neighbors (int, optional), # For split_approach 'saint', 'neighbor'
        return_nx (bool, optional),
        verbose (bool, optional)
    """
    data = load_graph(place,
                      feature_fields=all_feature_fields,
                      force_connected=True,
                      approach='dual',
                      clean=clean,
                      dist=dist,
                      cat_fields=cat_feats,
                      target_field=target_field,
                      verbose=True)
    if target_field == 'accident_count':
        data[target_field] = convert_accident_counts(data[target_field], categorize)

    if 'degree' in include_feats:
        add_deg_feats = True
        include_feats = remove_item(include_feats, 'degree')
    
    for cat_feat in cat_feats:
        include_feats = remove_item(include_feats, cat_feat)
        # Convert to categorical
        include_feats += [attr for attr in data.node_attrs if attr.startswith(cat_feat)]
    
    # Reduce data.x to only include_feats if desired
    node_attrs = data.node_attrs
    if include_feats is not None and include_feats != node_attrs:
        feat_idx = torch.tensor([node_attrs.index(feat) for feat in include_feats])
        feat_idx = feat_idx.to(data.x.device)
        data.num_nodes = data.x.size(0)
        data.x = torch.index_select(data.x, 1, feat_idx).float() \
                if len(feat_idx) > 0 else None

    if torch.is_tensor(data[target_field][0]):
        if data[target_field][0].dim() > 0:
            # Multiclass classification
            print(f'Classifying {target_field}')
            data.num_classes = len(data[target_field][0])
            counts = dict()
            for i in data[target_field]:
                key = i.sum().item()
                counts[key] = counts.get(key, 0) + 1
            data.loss_weights = torch.tensor([len(data[target_field]) / (2 * counts[i]) for i in range(data.num_classes)])
        else:
            # Regression
            print(f'Regression on {target_field}')
            data.num_classes = 1
        data.y = torch.stack(data[target_field])
    else:
        # Non-numeric => single class classification, relabel classes
        print(f'Classifying {target_field}')
        enc = LabelEncoder()
        target_data = np.array(data[target_field])
        enc_data = enc.fit_transform(target_data)
        data.num_classes = len(enc.classes_)
        data.y = torch.as_tensor(enc_data)
        print(f'Encoded target into the following classes: {enc.classes_}')
        counts = dict()
        for i in data.y:
            i = i.item()
            counts[i] = counts.get(i, 0) + 1
        data.loss_weights = torch.tensor([len(data.y) / (2 * counts[i]) for i in range(data.num_classes)])
        data.classes = enc.classes_
    
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.LocalDegreeProfile() if add_deg_feats else (lambda x: x),
        T.ToDevice(device) if split_approach != 'saint' else (lambda x: x),
        T.RandomNodeSplit(num_val=0.2, num_test=0)
    ])
    g = transform(data)
    
    # Loading graphs into PyG DataLoader
    if split_approach == 'cluster':
        if num_parts is None:
            num_parts = batch_size
            batch_size = 1
        cluster_data = ClusterData(g, num_parts=num_parts, log=verbose)
        loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
        test_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=False)
        g.split_loaders = False
    # elif split_approach == 'imbalanced':
    #     sampler = ImbalancedSampler(g, input_nodes=g.train_mask)
    #     loader = NeighborLoader(g, input_nodes=g.train_mask,
    #                     batch_size=64, num_neighbors=[-1, -1],
    #                     sampler=sampler, ...)
    elif split_approach == 'neighbor':
        kwargs = {'batch_size': batch_size, 'num_workers': 6, 'persistent_workers': True}
        loader = NeighborLoader(g, num_neighbors=[num_neighbors] * graph_length, shuffle=True,
                                  input_nodes=g.train_mask, **kwargs)
        test_loader = NeighborLoader(g, num_neighbors=[num_neighbors] * graph_length,
                                input_nodes=g.val_mask, **kwargs)
        g.split_loaders = True
    elif split_approach == 'saint':
        loader = GraphSAINTRandomWalkSampler(g, batch_size=batch_size, walk_length=graph_length,
                                             num_workers=6, shuffle=True)
        test_loader = GraphSAINTRandomWalkSampler(g, batch_size=batch_size, walk_length=graph_length,
                                             num_workers=6)
        g.split_loaders = False
    elif split_approach == 'khop':
        loader = ShaDowKHopSampler(g, depth=graph_length, num_neighbors=num_neighbors,
                                   node_idx=data.train_mask, batch_size=batch_size, shuffle=True)
        test_loader = ShaDowKHopSampler(g, depth=graph_length, num_neighbors=num_neighbors,
                                        node_idx=data.val_mask, batch_size=batch_size)
        g.split_loaders = True
    else:
        # Return loader with just the single graph
        loader = DataLoader([g], batch_size=1, shuffle=False)
        test_loader = loader
    return g, loader, test_loader

def print_naive_metrics(data, criteria, mask=True, multiclass=False):
    metric_dict = {crit.__name__: 0 for crit in criteria}
    target = (data.y[data.val_mask] if mask else data.y).cpu()
    naive_value = torch.mode(target, dim=0)[0]
    pred = torch.stack([naive_value for _ in target])
    
    if multiclass:
        target = target.sum(dim=1)
        pred = pred.sum(dim=1)
    for criterion in criteria:
        if criterion == f1_score:
            metric = criterion(target, pred, average='weighted')
        else:
            metric = criterion(target, pred)
        metric_dict[criterion.__name__] += metric
    print(f'{"Val " if mask else ""}Metrics when predicting all == {naive_value}:')
    print(metric_dict)

def train(model, loader, optimizer, criterion, masked=True):
    model.train()
    total_loss = 0
    num_samples = 0
    for data in loader:  # Iterate over each mini-batch.
        if data.x.device != device:
            data = data.to(device)
        out = model(data.x.float(), data.edge_index).squeeze(1)  # Perform a single forward pass.
        mask = data.train_mask if masked else torch.ones_like(data.y)
        loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients
        
        batch_num_samples = len(out[data.train_mask])
        total_loss += loss.item() * batch_num_samples
        num_samples += batch_num_samples
    return total_loss / num_samples 

def test_multiclass(model, loader, criteria, **_):
    model.eval()
    num_samples = 0
    metric_dict = {str(crit): 0 for crit in criteria}
    for data in loader:
        out = model(data.x.float(), data.edge_index).squeeze(1)
        
        # Get first index where value is less than threshold
        out = (out <= THRESHOLD).float().cpu()
        idx = torch.arange(out.shape[1], 0, -1)
        pred = torch.argmax(out * idx, 1, keepdim=True)

        val_y = data.y[data.val_mask].sum(dim=1).cpu()
        val_pred = pred[data.val_mask].detach().cpu()
        batch_num_samples = len(val_y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for criterion in criteria:
                if criterion == f1_score:
                    metric = criterion(val_y, val_pred, average='weighted')
                else:
                    metric = criterion(val_y, val_pred)
                metric_dict[str(criterion)] += metric * batch_num_samples
        num_samples += batch_num_samples
    
    for crit in metric_dict:
        metric_dict[crit] /= num_samples
    return list(metric_dict.values())

def test(model, loader, criteria, exp=False, masked=True):
    model.eval()
    num_samples = 0
    metric_dict = {str(crit): 0 for crit in criteria}
    for data in loader:
        if data.x.device != device:
            data = data.to(device)
        out = model(data.x.float(), data.edge_index).squeeze(1)
        if out.dim() > 1:
            pred = out.argmax(dim=1)  # Use the class with highest probability.
        else:
            pred = torch.exp(out) if exp else out # Regression: take the exponential if using poisson
        mask = data.val_mask if masked else torch.ones_like(data.y)
        val_y = data.y[mask].cpu()
        val_pred = pred[mask].detach().cpu()
        batch_num_samples = len(val_y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for criterion in criteria:
                if criterion == f1_score:
                    metric = criterion(val_y, val_pred, average='weighted')
                else:
                    metric = criterion(val_y, val_pred)
                metric_dict[str(criterion)] += metric * batch_num_samples
        num_samples += batch_num_samples
    
    for crit in metric_dict:
        metric_dict[crit] /= num_samples
    return list(metric_dict.values())


def run(
    place,
    gdf=None,
    loader = None,
    target_field = 'meridian_class',
    data_process_args = {
        'include_feats': rank_fields,
        'split_approach': 'cluster'
    },
    model_args = {
        'model_type': 'gat',
        'hidden_channels': 10,
        'num_layers': 2,
    },
    seed=42,
    num_iter=5,
    epochs=100,
    print_every=1,
    lr=0.001,
    schedule_lr=True,
    reg_loss='mse',
    criteria_names=None
):
    """
        Trainer for batched graph train-test split.
        
        Args:
        model_args (dict): Keyword arguments to modify the model including:
            out_channels (int): Latent variable dimension.
                (default: 10)
            model_type (str, optional): Type of gnn layer to use in the encoder.
                (default: 'gat')
            distmult (bool, optional): To set the decoder to DistMult. Defaults 
            to inner product.
                (default: False)
            linear (bool, optional): (default: False)
            variational (bool, optional): (default False)
            + Parameters to pass to torch's BasicGNN models (jk, num_layers, etc.)
    """
    if seed:
        seed_everything(seed)
    if loader is None:
        g, loader, test_loader = load_data(place, target_field,
                              verbose=True,
                              **data_process_args)
    
    in_channels = g.num_node_features
    out_channels = g.num_classes
    masked = not g.split_loaders
    model_args['in_channels'] = in_channels
    model_args['out_channels'] = out_channels
    
    # Criterion setting
    if out_channels == 1: 
        # Regression
        criterion = reg_loss_fns[reg_loss]
        test_criteria_names = ['MAE', 'Macro MAE', 'R2']
    else:
        # Classification
        if 'loss_weights' in g:
            print(f'CE Loss with weights {g.loss_weights}')
            criterion = torch.nn.CrossEntropyLoss(weight=g.loss_weights.to(device))
        else:
            criterion = torch.nn.CrossEntropyLoss()
        test_criteria_names = ['Accuracy', 'Macro Recall', 'F1', 'MCC']
    criteria_names = test_criteria_names if criteria_names is None else criteria_names
    test_criteria = [criteria_fns[name] for name in criteria_names]
    
    # Change test function if multiclass classification
    multiclass = 'categorize' in data_process_args and data_process_args['categorize'] == 'multiclass'
    test_fn = test_multiclass if multiclass else test
    
    print_naive_metrics(g, test_criteria, multiclass=multiclass)
    print_naive_metrics(g, test_criteria, multiclass=multiclass, mask=True)
    
    
    # Logging
    results = []
    models = []
    
    for iter_no in range(1, num_iter + 1):
        # Initialize a new model every iteration
        model_args_ = {i: model_args[i] for i in model_args if i!='model_type'}
        model = init_gnn_model(model_args['model_type'], **model_args_)
        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if schedule_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, 
                                                                   threshold=0.001, factor=0.5,verbose=True)
        
        result_dict = {'Train Loss': [], **{crit:[] for crit in test_criteria_names}}
        print(f'Starting iteration {iter_no} with model {model_args} ({num_params} params)')
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        start_time = time.time()
        epoch_start_time = start_time
        for epoch in range(1, epochs + 1):
            loss = train(model, loader, optimizer, criterion, masked=masked)
            metrics = test_fn(model, test_loader, test_criteria,
                              exp=(reg_loss == 'poisson'), masked=masked)
            
            # Logging
            result_dict['Train Loss'].append(loss)
            for idx, metric in enumerate(metrics):
                result_dict[test_criteria_names[idx]].append(metric)
             
            if schedule_lr:
                # Reduce LR if the target metric decreased
                # Assumes the target metric is the last metric in `test_criteria`
                scheduler.step(metrics[-1])
                
            if epoch % print_every == 0:
                sec_per_epoch = (time.time() - epoch_start_time) / print_every
                epoch_start_time = time.time()
                epoch_feed = (f'Epoch {epoch:03d} ({sec_per_epoch:.2f}s/epoch): '
                f'Train Loss: {loss:.3f}')
                for i, crit in enumerate(test_criteria_names):
                    epoch_feed += f', {crit}: {metrics[i]:.3f}'
                print(epoch_feed)
        
        result_df = pd.DataFrame.from_records([{
            key: result_dict[key][-1]
            for key in result_dict
            if len(result_dict[key]) > 0
        }])
        
        sec_per_epoch = (time.time() - start_time) / epochs
        print(f'Iteration {iter_no} done, averaged {sec_per_epoch:.3f}s per epoch. Results:')
        display(result_df)
        result_dict['sec_per_epoch'] = sec_per_epoch
        result_dict['model_details'] = model.__repr__()
        result_dict['model_parameters'] = num_params
        results.append(result_dict)
        models.append(model)
    
    return models, results
