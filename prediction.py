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
import matplotlib.pyplot as plt
from IPython.display import display
from numbers import Number
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, ClusterData, ClusterLoader, NeighborLoader, GraphSAINTRandomWalkSampler, ShaDowKHopSampler
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.losses import coral_loss

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, r2_score, mean_absolute_error, mean_squared_error
from imblearn.metrics import macro_averaged_mean_absolute_error
from sklearn.preprocessing import LabelEncoder, FunctionTransformer

from models.init_model import init_gnn_model, CoralGNN
from utils.early_stopping import EarlyStopping
from utils.utils import remove_item
from utils.seed import seed_everything
from utils.normalize_features import NormalizeFeatures
from utils.load_geodata import load_graph, load_gdf
from utils.constants import rank_fields, all_feature_fields, geom_feats, ignore_non_accident_field

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5
NUM_WORKERS = 4

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
    'MSE': mean_squared_error,
    'RMSE': partial(mean_squared_error, squared=False),
    'Macro MAE': macro_mae,
    'R2': r2_score,
    'Accuracy': accuracy_score,
    'Macro Recall': balanced_accuracy_score,
    'F1': f1_score,
    'Micro F1': partial(f1_score, average='micro'),
    'Weighted F1': partial(f1_score, average='weighted'),
    'MCC': matthews_corrcoef
}

def poissonLoss(xbeta, y):
    """Custom loss function for Poisson model."""
    loss=torch.mean(torch.exp(xbeta)-y*xbeta)
    return loss


def convert_accident_counts(tensor_data, categorize, bins=[0, 1, 2, 3, 5, np.inf]):
    if categorize == 'classification':
        labels = ['No Accidents', 'One Accident', 'Two Accidents', 'Few Accidents', 'Many Accidents']
    elif categorize == 'regression':
        labels = [0., 1., 2., 3., 4.]
        labels = [torch.tensor(label) for label in labels]  
    elif categorize == 'multiclass':
        labels = [0, 1, 2, 3, 4]
        labels = [torch.tensor(label) for label in labels]
    else:
        return tensor_data
    
    print(f'Categorizing accident_count')
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
    scaler='power',
    clean=True,
    dist=50,
    agg='sum',
    split=0.8,
    batch_size=64,
    split_approach=None,
    num_parts=None, # For split_approach='cluster'
    num_neighbors=-1, # For split_approach 'saint', 'neighbor'
    graph_length=2,
    return_nx=False,
    verbose=False
):
    """
    Loads and prepares the graph of roads from `place` for node classification.
    
    Args:
        place (str): Name of Local Authority, or 'No Bounds' for full graph,
        target_field (str): Name of target attribute,
        categorize (str, optional): Classification strategy
            Options: ['classification', 'multiclass', 'regression'],
        include_feats (list, optional),
        cat_feats (list, optional),
        add_deg_feats (bool, optional),
        scaler (str, optional): Type of Normalization scaler to use
            Options: ['sum', 'minmax', 'maxabs', 'robust', 'power']
        clean (bool, optional),
        dist (int, optional),
        agg (str, optional),
        split (float, optional): Ratio of data to put into train,
        batch_size (int, optional),
        split_approach (str, optional),
        num_parts (int, optional) # For split_approach='cluster'
        num_neighbors (int, optional), # For split_approach 'saint', 'neighbor'
        graph_length (int, optional): Size of subgraph in hops
        return_nx (bool, optional),
        verbose (bool, optional)
    """
    # Only graphs with meridian_class have been saved, so forcibly load that graph and remove the feats later
    cat_fields = ['meridian_class'] if target_field == 'accident_count' else cat_feats
    clean_agg = 'sum' if target_field == 'meridian_class' else agg
    data = load_graph(place,
                      feature_fields=all_feature_fields,
                      force_connected=True,
                      approach='dual',
                      clean=clean,
                      clean_agg=clean_agg,
                      dist=dist,
                      cat_fields=cat_fields,
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
        if categorize == 'multiclass':
            # Multiclass classification
            if verbose:
                print(f'Classifying {target_field}')
            data.num_classes = 5
            data.label = levels_from_labelbatch(data[target_field], num_classes=data.num_classes)
            # data.num_classes = len(data[target_field][0])
            # counts = dict()
            # for i in data[target_field]:
            #     key = i.sum().item()
            #     counts[key] = counts.get(key, 0) + 1
            # data.loss_weights = torch.tensor([len(data[target_field]) / (2 * counts[i]) for i in range(data.num_classes)])
        else:
            # Regression
            if verbose:
                print(f'Regression on {target_field}')
            data.num_classes = 1
        data.y = torch.stack(data[target_field])
    else:
        # Non-numeric => single class classification, relabel classes
        enc = LabelEncoder()
        target_data = np.array(data[target_field])
        enc_data = enc.fit_transform(target_data)
        data.num_classes = len(enc.classes_)
        data.y = torch.as_tensor(enc_data)
        if verbose:
            print(f'Encoded {target_field} into the following classes: {enc.classes_}')
        counts = dict()
        for i in data.y:
            i = i.item()
            counts[i] = counts.get(i, 0) + 1
        data.loss_weights = torch.tensor([len(data.y) / (2 * counts[i]) for i in range(data.num_classes)])
        data.classes = enc.classes_
    
    transform = T.Compose([
        NormalizeFeatures(scaler=scaler),
        T.LocalDegreeProfile() if add_deg_feats else (lambda x: x),
        T.ToDevice(device) if split_approach != 'saint' else (lambda x: x),
        T.RandomNodeSplit(num_val=(1 - split), num_test=0)
    ])
    g = transform(data)
    
    # Loading graphs into PyG DataLoader
    kwargs = {'batch_size': batch_size, 'num_workers': NUM_WORKERS, 'persistent_workers': True} # For samplers
    if split_approach == 'cluster':
        if num_parts is None:
            num_parts = batch_size
            batch_size = 1
        start_time = time.time()
        cluster_data = ClusterData(g, num_parts=num_parts, log=verbose)
        loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
        test_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=False)
        loader_time = time.time() - start_time
        if verbose:
            print(f'Cluster Loader with {num_parts} clusters, {batch_size} graphs per batch (Time taken: {loader_time:.3f}s)')
        g.split_loaders = False
    # elif split_approach == 'imbalanced':
    #     sampler = ImbalancedSampler(g, input_nodes=g.train_mask)
    #     loader = NeighborLoader(g, input_nodes=g.train_mask,
    #                     batch_size=64, num_neighbors=[-1, -1],
    #                     sampler=sampler, ...)
    elif split_approach == 'neighbor':
        start_time = time.time()
        loader = NeighborLoader(g, num_neighbors=[num_neighbors] * graph_length, shuffle=True,
                                  input_nodes=g.train_mask, **kwargs)
        test_loader = NeighborLoader(g, num_neighbors=[num_neighbors] * graph_length, shuffle=False,
                                input_nodes=g.val_mask, **kwargs)
        loader_time = time.time() - start_time 
        if verbose:
            print(f'Neighbor Loader with {num_neighbors} neighbors over {graph_length} steps, {batch_size} samples per batch (Time taken: {loader_time:.3f}s)')
        g.split_loaders = True
    elif split_approach == 'saint':
        if verbose:
            print(f'SAINT Random Walk Loader with {graph_length} steps, {batch_size} samples per batch')
        loader = GraphSAINTRandomWalkSampler(g, walk_length=graph_length, **kwargs)
        test_loader = loader
        g.split_loaders = False
    # elif split_approach == 'khop':
    #     loader = ShaDowKHopSampler(g, depth=graph_length, num_neighbors=num_neighbors,
    #                                node_idx=data.train_mask, batch_size=batch_size, shuffle=True)
    #     test_loader = ShaDowKHopSampler(g, depth=graph_length, num_neighbors=num_neighbors,
    #                                     node_idx=data.val_mask, batch_size=batch_size)
    #     g.split_loaders = True
    else:
        # Return loader with just the single graph
        if verbose:
            print('No split_approach specified, defaulting to full batch...')
        loader = DataLoader([g], batch_size=1, shuffle=False)
        test_loader = loader
        g.split_loaders = False
    return g, loader, test_loader

def print_naive_metrics(data, criteria, names, mask=True):
    metric_dict = {names[idx]: 0 for idx, crit in enumerate(criteria)}
    target = (data.y[data.val_mask] if mask else data.y).cpu()
    naive_value = torch.mode(target, dim=0)[0]
    pred = torch.stack([naive_value for _ in target])

    for idx, criterion in enumerate(criteria):
        metric = criterion(target, pred)
        metric_dict[names[idx]] += metric
    print(f'{"Val " if mask else ""}Metrics when predicting all == {naive_value}:')
    print(*(f'{metric}: {metric_dict[metric]:.3f}' for metric in metric_dict))

def train(model, loader, optimizer, criterion, masked=True, coral=False):
    model.train()
    total_loss = 0
    num_samples = 0
    for data in loader:  # Iterate over each mini-batch.
        if not data.x.is_cuda:
            data = data.to(device)
        mask = data.train_mask if masked else torch.ones_like(data.y).bool()
        if coral:
            # levels = levels_from_labelbatch(data.y[mask], num_classes=model.out_channels)
            # levels = levels.to(device)
            logits, _ = model(data.x.float(), data.edge_index)
            #### CORAL loss 
            loss = coral_loss(logits[mask], data.label[mask])
        else:
            out = model(data.x.float(), data.edge_index).squeeze(1)  # Perform a single forward pass.
            loss = criterion(out[mask], data.y[mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients
        
        batch_num_samples = len(data.y[mask])
        total_loss += loss.item() * batch_num_samples
        num_samples += batch_num_samples
    return total_loss / num_samples 


def test(model, loader, loss_fn, criteria, exp=False, masked=True, coral=False):
    model.eval()
    total_loss, num_samples = 0, 0
    metric_dict = {str(crit): 0 for crit in criteria}
    for data in loader:
        if not data.x.is_cuda:
            data = data.to(device)
        
        # Compute validation loss
        mask = data.val_mask if masked else torch.ones_like(data.y).bool()
        val_y = data.y[mask].cpu()
        batch_num_samples = len(val_y)
        
        # Make predictions
        if coral:
            # levels = levels_from_labelbatch(val_y, num_classes=model.out_channels)
            # levels = levels.to(device)
            #### CORAL loss 
            logits, probas = model(data.x.float(), data.edge_index)
            val_loss = coral_loss(logits[mask], data.label[mask])
            pred = proba_to_label(probas).float()
        else:
            out = model(data.x.float(), data.edge_index).squeeze(1)
            val_loss = loss_fn(out[mask], data.y[mask])
        
            if out.dim() > 1:
                pred = out.argmax(dim=1)  # Use the class with highest probability.
            else:
                pred = torch.exp(out) if exp else out # Regression: take the exponential if using poisson
        total_loss += val_loss.item() * batch_num_samples
        val_pred = pred[mask].detach().cpu()
        
        # Compute metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for criterion in criteria:
                metric = criterion(val_y, val_pred)
                metric_dict[str(criterion)] += metric * batch_num_samples
        
        num_samples += batch_num_samples
    
    for crit in metric_dict:
        metric_dict[crit] /= num_samples
    return total_loss / num_samples, list(metric_dict.values())


def run(
    place='No Bounds',
    inductive_place=None,
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
    schedule_lr=False,
    test_inductive=False,
    early_stopping=False,
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
    assert all (key in model_args for key in ('hidden_channels', 'num_layers'))
    if seed:
        seed_everything(seed)
    if loader is None:
        if 'graph_length' not in data_process_args:
            data_process_args['graph_length'] = model_args['num_layers']
        g, loader, test_loader = load_data(place, 
                                           target_field,
                                           verbose=True, 
                                           **data_process_args)
    if inductive_place is not None:
        print('Loading graph data for inductive testing...')
        inductive_g, inductive_loader, _ = load_data(inductive_place, 
                                            target_field, 
                                            split=0,
                                            verbose=True, 
                                            **data_process_args)
    
    in_channels = g.num_node_features
    out_channels = g.num_classes
    masked = not g.split_loaders
    model_args['in_channels'] = in_channels
    model_args['out_channels'] = out_channels
    
    # Change test function if multiclass classification
    multiclass = 'categorize' in data_process_args and data_process_args['categorize'] == 'multiclass'
    
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
        test_criteria_names = ['Accuracy', 'Macro Recall', 'Micro F1', 'MCC', 'Weighted F1']
    test_criteria_names = criteria_names if criteria_names is not None else test_criteria_names
    test_criteria = [criteria_fns[name] for name in test_criteria_names]
    
    # Print the achieved metrics of a naive classifier
    print_naive_metrics(g, test_criteria, test_criteria_names, mask=masked)
    if inductive_place is not None:
        print('Inductive:')
        print_naive_metrics(inductive_g, test_criteria, test_criteria_names, mask=False)
    
    # Logging
    results = []
    models = []
    
    for iter_no in range(1, num_iter + 1):
        # Initialize a new model every iteration
        if multiclass:
            model = CoralGNN(**model_args)
        else:
            model_args_ = {i: model_args[i] for i in model_args if i!='model_type'}
            model = init_gnn_model(model_args['model_type'], **model_args_)
        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if schedule_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, 
                                                                   threshold=0.001, factor=0.5,verbose=True)
        if early_stopping:
            early_stop = False
            early_stopper = EarlyStopping(patience=10, verbose=True)
        
        result_dict = {'Train Loss': [], 'Val Loss': [], **{crit:[] for crit in test_criteria_names}}
    
        print(f'Starting iteration {iter_no} with model {model_args} ({num_params} params)')
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        start_time = time.time()
        epoch_start_time = start_time
        for epoch in range(1, epochs + 1):
            loss = train(model, loader, optimizer, criterion, masked=masked, coral=multiclass)
            
            val_loss, metrics = test(model, test_loader, criterion, test_criteria,
                                     masked=masked, coral=multiclass)
            # Logging
            result_dict['Train Loss'].append(loss)
            result_dict['Val Loss'].append(val_loss)
            for idx, metric in enumerate(metrics):
                result_dict[test_criteria_names[idx]].append(metric)
                
            if epoch % print_every == 0:
                sec_per_epoch = (time.time() - epoch_start_time) / print_every
                epoch_feed = (f'Epoch {epoch:03d} ({sec_per_epoch:.2f}s/epoch): '
                f'Train Loss: {loss:.3f}, Val Loss: {val_loss:.3f}')
                for i, crit in enumerate(test_criteria_names):
                    epoch_feed += f', {crit}: {metrics[i]:.3f}'
                print(epoch_feed)
                epoch_start_time = time.time()
            
            if schedule_lr:
                # Reduce LR if the val loss increased
                scheduler.step(val_loss)
            if early_stopping:
                early_stopper(val_loss, model)
                if early_stopper.early_stop:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        sec_per_epoch = (time.time() - start_time) / epochs
        
        result_df = pd.DataFrame.from_records([{
            key: result_dict[key][-1]
            for key in result_dict
            if len(result_dict[key]) > 0
        }])
        
        # Obtain inductive metrics post-training
        if inductive_place is not None:
            _, inductive_metrics = test(model, inductive_loader, criterion,
                                        test_criteria, masked=False, coral=multiclass)
            for idx, metric in enumerate(inductive_metrics):
                inductive_crit_name = f'Inductive {test_criteria_names[idx]}'
                result_dict[inductive_crit_name] = metric
                result_df[inductive_crit_name] = metric
        
        print(f'Iteration {iter_no} done, averaged {sec_per_epoch:.3f}s per epoch. Results:')
        display(result_df)
        result_dict['sec_per_epoch'] = sec_per_epoch
        result_dict['model_details'] = model.__repr__()
        result_dict['model_parameters'] = num_params
        results.append(result_dict)
        
        if early_stopping:
            model = early_stopper.load_checkpoint(model)
        models.append(model)
    
    return models, results
