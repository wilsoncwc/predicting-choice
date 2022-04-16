import copy
import math
import time
import numpy as np

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader   
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score

from models import init_model
from utils.seed import seed_everything
from utils.constants import project_root, dataset_root
from utils.constants import feats, rank_fields, metric_dict, default_model
from utils.constants import GLOBAL_THRESHOLD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    auc_tot = 0
    ap_tot = 0
    for data in loader:
        z = model.encode(data.x, data.edge_index)
        auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        auc_tot += auc
        ap_tot += ap
    return auc_tot / len(loader), ap_tot / len(loader)


def run(
    dataset,
    data_process_args={},
    model_args=default_model,
    seed=42,
    num_iter=5,
    save_best_model=False,
    save_best_model_metric='test_ap',
    output_tb=False,
    tag_tb='',
    epochs=1000,
    print_every=10,
    lr=0.01,
    schedule_lr=False,
    beta=1 # only used for GVAE
):
    """
        Trainer for batched graph train-test split.
        data_process_args: dict of optional keywords including:
            split, hold_out_edge_ratio, batch_size, include_feats, add_deg_feats
        model_args: dict of optional keywords including:
            out_channels: int - latent variable dimension,
            model_type: str - type of gnn layer to use in the encoder,
            distmult: bool - set to True for DistMult decoder, else default inner product
            linear: bool - flag
            variational: bool - flag
            + parameters to pass to torch's BasicGNN models (jk, num_layers, etc.)
    """
    if seed:
        seed_everything(seed)
    if type(dataset) is tuple:
        # Assume already processed
        train_dataset, test_dataset, train_loader, test_loader = dataset
    else:
        train_dataset, test_dataset, train_loader, test_loader = process_dataset(dataset, **data_process_args)
    in_channels = train_dataset[0].num_node_features
    model_args['in_channels'] = in_channels
    
    model_args['variational'] = model_args['variational'] if 'variational' in model_args else False
    mod = 'V' if model_args['variational'] else ''
    mod += 'L' if 'linear' in model_args and model_args['linear'] else ''
    dist = '-dist' if 'distmult' in model_args and model_args['distmult'] else ''
    add_deg_feats = data_process_args['add_deg_feats']
    include_feats = data_process_args['include_feats']
    tag_tb = f'{"deg+" if add_deg_feats else ""}{len(include_feats)}_{tag_tb}'
    model_tag = f'G{mod}AE_{model_args["out_channels"]}d_{model_args["model_type"]}{dist}'
    run_str = f'{model_tag}_{epochs}epochs_{lr}lr_{tag_tb}feats'
    
    results = []
    models = []
    start_time = time.time()
    for i in range(1, num_iter + 1):
        print(f'Running iteration {i} of expt {run_str}')
        model = init_model(**model_args)
        model = model.to(device)
        model.data_process_args = data_process_args
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if schedule_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
        result_dict = {metric: [] for metric in metric_dict}
        for epoch in range(1, epochs + 1):
            total_loss, recon_loss, kl_loss = train(model, optimizer, train_loader, model_args['variational'], beta)
            train_auc, train_ap = test(model, train_dataset)
            test_auc, test_ap = test(model, test_dataset)
            if schedule_lr:
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

def test_model_inductive(processed_dataset, model):
    """
    Obtain inductive metrics for models trained on single LA
    """
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

def run_single(places, full_dataset, run_args={}, save_path=''):
    """
        Trainer for individual local authorities.
        Trains a model for each place in the list input 'places', obtaining transductive metrics.
        Tests each model on all other local authorities to obtain inductive metrics, and saves the average.
        'run_args' are passed to the main 'run' function, see above for possible arguments.
    """
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
            metrics = test_model_inductive(post_run_test_dataset, model)
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