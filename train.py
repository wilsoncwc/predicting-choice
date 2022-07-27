import copy
import math
import time
import numpy as np
import pandas as pd

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score

from models.init_model import init_model
from utils import *
from utils.constants import project_root, dataset_root
from utils.constants import rank_fields, metric_dict, default_model
from utils.constants import GLOBAL_THRESHOLD
from utils.normalize_features import NormalizeFeatures
from utils.custom_link_split import CustomLinkSplit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_dataset(full_dataset, split=0.8, hold_out_edge_ratio=0.2,
                    neg_sampling_ratio=-1, batch_size=64, include_feats=[], scaler='quantile-normal',
                    feat_scaler_dict={}, add_deg_feats=False, deg_after_split=False, verbose=False):
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
    
    # Adds Local Degree Profile
    if 'degree' in include_feats:
        add_deg_feats = True
        include_feats = remove_item(include_feats, 'degree')
        
    # Remove non included features
    for data in dataset:
        if include_feats != data.node_attrs:
            filter_feats(data, include_feats)
    
    # If no features are selected, inject degree profile as features
    # else, normalize the features
    neg_sampling_ratio = neg_sampling_ratio if neg_sampling_ratio >= 0 else 1 / hold_out_edge_ratio
    transform = T.Compose([
        NormalizeFeatures(scaler=scaler, feat_scaler_dict=feat_scaler_dict),
        T.LocalDegreeProfile() if add_deg_feats and not deg_after_split else (lambda x: x),
        T.ToDevice(device),
        CustomLinkSplit(num_val=0, num_test=0, disjoint_train_ratio=hold_out_edge_ratio,
                          neg_sampling_ratio=neg_sampling_ratio, is_undirected=True)
    ])

    train_dataset = [transform(dataset[i])[0] for i in train_idx]
    test_dataset = [transform(dataset[i])[0] for i in test_idx]
    if deg_after_split and add_deg_feats:
        deg_xform = T.Compose([T.LocalDegreeProfile(), T.ToDevice(device)])
        train_dataset = [deg_xform(data) for data in train_dataset]
        test_dataset = [deg_xform(data) for data in test_dataset]

    if verbose:
        positive = sum([len(data.pos_edge_label_index) for data in test_dataset])
        negative = sum([len(data.neg_edge_label_index) for data in test_dataset])
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(test_dataset)}')
        print(f'Positive rate: {positive / (positive + negative)}')

    # Load graphs into dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) \
        if len(train_dataset) > 0 else []
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False) \
        if len(test_dataset) > 0 else []
    if verbose:
        for step, data in enumerate(train_loader):
            print(f'Step {step + 1}:')
            print('=======')
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print(data)
            print()

    return train_dataset, test_dataset, train_loader, test_loader


def train(model, optimizer, loader, variational, beta=1):
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
            kl_loss = (beta / data.num_nodes) * model.kl_loss()
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
    auc_sum, ap_sum, count = 0, 0, 0
    for data in loader:
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        auc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
        ap = average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
        
        num_graphs = data.num_graphs if isinstance(data, Batch) else 1
        auc_sum += auc * num_graphs
        ap_sum += ap * num_graphs
        count += num_graphs
    return auc_sum / count, ap_sum / count

@torch.no_grad()
def test_enhanced(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    f1 = model.test_enhanced(data)
    return f1

@torch.no_grad()
def test_curve(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    roc, pr = model.test_curves(z, data.pos_edge_label_index, data.neg_edge_label_index)
    return roc, pr
    
def run(
    dataset,
    data_process_args={},
    model_args=default_model,
    seed=42,
    test_inductive=True,
    num_iter=5,
    save_best_model=False,
    return_best_model=False,
    save_best_model_metric='train_ap',
    output_tb=False,
    tag_tb='',
    epochs=1000,
    print_every=10,
    lr=0.01,
    schedule_lr=False,
    min_lr=0.00001,
    beta=1, # only used for GVAE
    verbose=True
):
    """
        Trainer for batched graph train-test split.
        
        Args:
        data_process_args (dict): Optional keywords for data preprocessing/batching
            split, hold_out_edge_ratio, batch_size, include_feats, add_deg_feats
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
    if type(dataset) is tuple:
        # Assume already processed via process_dataset
        train_dataset, test_dataset, train_loader, test_loader = dataset
    else:
        train_dataset, test_dataset, train_loader, test_loader = process_dataset(dataset, **data_process_args)
    in_channels = train_dataset[0].num_node_features
    model_args['in_channels'] = in_channels

    # Format the output filename to which models and results are saved
    model_args['variational'] = model_args['variational'] if 'variational' in model_args else False
#     mod = 'V' if model_args['variational'] else ''
#     mod += 'L' if 'linear' in model_args and model_args['linear'] else ''
#     dist = '-dist' if 'distmult' in model_args and model_args['distmult'] else ''
    
#     add_deg_feats = data_process_args.get('add_deg_feats', False)
#     include_feats = data_process_args.get('include_feats', [])
#     tag_tb = f'{"deg+" if add_deg_feats else ""}{len(include_feats)}_{tag_tb}'
#     model_tag = f'G{mod}AE_{model_args["out_channels"]}d_{model_args["model_type"]}{dist}'
    run_str = str(list({**data_process_args}.values()))
        
    # Logging
    results = []
    models = []
    
    for i in range(1, num_iter + 1):
        if verbose:
            print(f'Running iteration {i} of expt {run_str}')
            
        if output_tb:
            # Output average run losses to tensorboard
            path = f'{project_root}/runs/{run_str}/run_{i}'
            if verbose:
                print(f'Writing to {path}')
            writer = SummaryWriter(path)
        start_time = time.time()
        epoch_start_time = start_time
        
        # Initialize a new model every iteration
        model = init_model(verbose=verbose, **model_args)
        model = model.to(device)
        model.data_process_args = data_process_args
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if schedule_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20, 
                                                                   threshold=0.001, factor=0.5, verbose=True)
    
        result_dict = {metric: [] for metric in metric_dict}
        if not test_inductive:
            result_dict.pop('test_auc')
            result_dict.pop('test_ap')
        if not model_args['variational']:
            result_dict.pop('recon_loss')
            result_dict.pop('kl_loss')
    
        for epoch in range(1, epochs + 1):
            total_loss, recon_loss, kl_loss = train(model, optimizer, train_loader, model_args['variational'], beta)
            # Transductive metrics
            train_auc, train_ap = test(model, train_dataset)
            
            # Logging
            result_dict['total_loss'].append(total_loss)
            if model_args['variational']:
                result_dict['recon_loss'].append(recon_loss)
                result_dict['kl_loss'].append(kl_loss)
            result_dict['train_auc'].append(train_auc)
            result_dict['train_ap'].append(train_ap)
            
            # Inductive metrics
            if test_inductive:
                test_auc, test_ap = test(model, test_dataset)
                result_dict['test_auc'].append(test_auc)
                result_dict['test_ap'].append(test_ap)
             
            if schedule_lr:
                # Reduce LR if the target metric decreased
                scheduler.step(result_dict[save_best_model_metric][-1])
                
            if verbose and epoch % print_every == 0:
                sec_per_epoch = (time.time() - epoch_start_time) / print_every
                epoch_start_time = time.time()
                epoch_feed = (f'Epoch {epoch:03d} ({sec_per_epoch:.2f}s/epoch): '
                f'Train AUC: {train_auc:.4f}, Train AP: {train_ap:.4f}')
                if test_inductive:
                    epoch_feed += f',Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}'
                print(epoch_feed)
                
            if output_tb:
                for metric in result_dict:
                    result = result_dict[metric][epoch - 1]
                    writer.add_scalar(metric_dict[metric], result, epoch - 1)
        
        result_df = pd.DataFrame.from_records([{
            key: result_dict[key][-1]
            for key in result_dict
            if len(result_dict[key]) > 0
        }])
        
        sec_per_epoch = (time.time() - start_time) / epochs
        if verbose:
            print(f'Iteration {i} done, averaged {sec_per_epoch:.3f}s per epoch. Results:')
            display(result_df)
        # roc_curve, pr_curve = test_curve(model, )
        # result_dict['roc_curve'] = roc_curve
        # result_dict['pr_curve'] = pr_curve
        result_dict['sec_per_epoch'] = sec_per_epoch
        result_dict['model_details'] = model.__repr__()
        results.append(result_dict)
        models.append(model)

    if output_tb:
        # Output average run losses to tensorboard
        path = f'{project_root}/runs/{run_str}'
        if verbose:
            print(f'Writing average results to {path}')
        writer = SummaryWriter(path)
        mean_result_dict = {}
        for metric in metric_dict:
            mean_result = np.mean([res[metric] for res in results if metric in res], axis=0)
            if type(mean_result) == list:
                mean_result_dict[metric] = mean_result[-1]
                for i, result in enumerate(mean_result):
                    writer.add_scalar(metric_dict[metric], result, i)
            elif type(mean_result) == float:
                mean_result_dict[metric] = mean_result
        
        # Log run and model parameters
        data_process_args['include_feats'] = ', '.join(data_process_args['include_feats'])
        hparams = {**data_process_args, **model_args}
        writer.add_hparams({key: str(hparams[key]) for key in hparams},
                           mean_result_dict)
    
    if save_best_model or return_best_model:
        results_best_metric = [res[save_best_model_metric][-1] for res in results]
        best_iteration = np.argmax(results_best_metric)
        if save_best_model:
            path = f'{project_root}/saved_models/{run_str}.pt'
            if verbose:
                print(f'Saving model to {path}')
            torch.save(models[best_iteration].state_dict(), path)
        if return_best_model:
            return models[best_iteration], results[best_iteration]
    return models, results

def test_model_inductive(processed_dataset, model):
    """
    Obtain inductive metrics for models trained on single LA
    """
    test_result_dict = {}
    for test_data in processed_dataset:
        # Obtain and save latent representation
        z = model.encode(test_data.x, test_data.edge_index)
        result_dict = {'z': z.detach().cpu()}
        
        # Create binary classification label
        pos_y = z.new_ones(test_data.pos_edge_label_index.size(1))
        neg_y = z.new_zeros(test_data.neg_edge_label_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        
        # Perform prediction
        pos_pred = model.decode(z, test_data.pos_edge_label_index)
        neg_pred = model.decode(z, test_data.neg_edge_label_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        # Pre-threshold metrics
        result_dict['auc'] = roc_auc_score(y, pred)
        result_dict['ap'] = average_precision_score(y ,pred)
        
        # Post-threshold metrics:
        thresholded_pred = (pred > GLOBAL_THRESHOLD).astype(float)
        result_dict['precision'] = precision_score(y, thresholded_pred)
        result_dict['recall'] = recall_score(y, thresholded_pred)
        result_dict['f1'] = f1_score(y, thresholded_pred)
        
        test_result_dict[test_data.place] = result_dict
    return test_result_dict

def run_all(dataset, run_args={}, save_path=''):
    output_dict = {}
    data_process_args = {
        'split': 1, # indicates ALL data goes into train dataset (0 test)
        'batch_size': run_args.pop('batch_size', 16),
        'add_deg_feats': run_args.pop('add_deg_feats', False),
        'deg_after_split': run_args.pop('deg_after_split', False),
        'include_feats': run_args.pop('include_feats', rank_fields)
    }
    
    data = process_dataset(dataset, **data_process_args)
    
    model, result = run(data,
                        data_process_args, 
                        test_inductive=False,
                        return_best_model=True,
                        **run_args)

    print('Model training ended. Computing metrics...')
    result = summarize_results([result])
    
    # Apply best performing model to every city
    metrics = test_model_inductive(data[0], model)
    combined = {**result, **metrics}
    if save_path:
        torch.save(combined, save_path)
    return result, metrics
        

def run_single(places, full_dataset, data_process_args={}, run_args={}, save_path='',
               summarize=True, only_transductive=False, enhanced=True):
    """
        Trainer for individual local authorities.
        Trains a model for each place in the list input 'places', obtaining transductive metrics.
        Tests each model on all other local authorities to obtain inductive metrics, and saves the average.
        'run_args' are passed to the main 'run' function, see above for possible arguments.
    """
    output_dict = {}
    data_process_args = {
        'split': 0, # indicates ALL data goes into test dataset (0 train)
        'batch_size': 32,
        'add_deg_feats': False,
        'deg_after_split': False,
        'include_feats': rank_fields,
        **data_process_args,
    }
    _, test_dataset, _, test_loader = process_dataset(full_dataset, verbose=False, **data_process_args)
    # post_run_test_dataset = process_dataset(full_dataset, verbose=False, split_labels=True, 
                                            # **data_process_args)[1]
    for idx, place in enumerate(places):
        print(f'Training model on SSx data from {place} ({idx + 1}/{len(places)})...')
        place_graph = next(data for data in test_dataset if data.place == place)
        train_dataset = [place_graph] # train on a single graph only
        train_loader = DataLoader(train_dataset, batch_size=1)
        dataset = train_dataset, test_dataset, train_loader, test_loader
        
        models, results = run(dataset, data_process_args, 
                              test_inductive=False, **run_args)
        
        print(f'Model training ended for {place} ({idx + 1}/{len(places)}). Computing metrics...')
        # Append final metrics of each run
        data = summarize_results(results) if summarize else {'results': results}
            
        if only_transductive:
            # Get filtered prediction metrics and auc/pr curves
            data.update({'filtered_f1': [], 'roc': [], 'pr': []})
            for model in models:
                if enhanced:
                    data['filtered_f1'].append(test_enhanced(model, place_graph))
                auc, ap = test(model, [place_graph])
                roc, pr = test_curve(model, place_graph) 
                data['roc'].append(roc)
                data['pr'].append(pr)
    
            # Only output test metrics for transductive setting
            output_dict[place] = data
            if save_path:
                print('Saving results...')
                torch.save(output_dict, save_path)
            continue
            
        # Test models on every city and compute average over models
        test_dict = {}
        for model in models:
            metrics = test_model_inductive(test_dataset, model)
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
            if result_place == place:
                # Only save the encoded representation of the LA used for training
                data['encoding'] = result_metrics['z']
            del result_metrics['z']
            
            for metric in result_metrics:
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
        data['inductive'] = test_dict
        
        output_dict[place] = data
        if save_path:
            print(f'Saving results to {save_path}')
            torch.save(output_dict, save_path)

    return output_dict