#!/usr/bin/env python
# coding: utf-8

# Set path to include project root so that modules can be directly imported
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from train import run, run_all
from utils.constants import *

def main():
    out_file = f'{dataset_root}/model_runs/link_pred_fixedsum_feat_runs.pt'
    dataset = torch.load(f'{dataset_root}/ssx_dataset_min.pt')
    result_dict = {}
    for var_args in [
        ['x', 'y'],
        km2_fields,
        km10_fields,
        km100_fields,
        choice_fields,
        integration_fields,
        nodecount_fields,
        ['choice2kmrank', 'choice10kmrank'],
        ['integration2kmrank', 'integration10kmrank'],
        rank_fields,
        unnorm_feature_fields,
        og_feature_fields,
    ]:
        model_args = {
            'model_type': 'gain',
            'out_channels': 10,
            'num_layers': 2,
        }
        proc_args = {
            'include_feats': var_args,
            'scaler': 'sum',
            'add_deg_feats': False,
            'batch_size': 8,
            'verbose': False
        }
        print(f'Testing {var_args}')
        models, results = run(dataset,
                            proc_args,
                            model_args,
                            num_iter=5,
                            lr=0.01,
                            epochs=500,
                            schedule_lr=True,
                            output_tb=False,
                            save_best_model=False)
        result_dict[str(var_args)] = results
        torch.save(result_dict, out_file)
        print(f'Saved results to {out_file}')

if __name__ == '__main__':
    main()
    
    
    
"""
        {'model_type': 'gcn', 'distmult': True},
        {'model_type': 'gat', 'distmult': True},
        # {'model_type': 'sage', 'distmult': True, 'aggr': 'min'},
        # {'model_type': 'sage', 'distmult': True, 'aggr': 'mean'},
        {'model_type': 'sage', 'distmult': True, 'aggr': 'max'},
        # {'model_type': 'sage', 'distmult': True, 'aggr': 'sum'},
        {'model_type': 'gin', 'distmult': True},
        {'model_type': 'gain', 'distmult': True},
        {'model_type': 'gcn', 'distmult': False},
        {'model_type': 'gat', 'distmult': False},
        # {'model_type': 'sage', 'distmult': False, 'aggr': 'min'},
        # {'model_type': 'sage', 'distmult': False, 'aggr': 'mean'},
        {'model_type': 'sage', 'distmult': False, 'aggr': 'max'},
        # {'model_type': 'sage', 'distmult': False, 'aggr': 'sum'},
        {'model_type': 'gin', 'distmult': False},
        {'model_type': 'gain', 'distmult': False},
        
        
'max',
'min',
'sum',
'mean',
'median',
'std'

result_dict[var_str] = results
dataset = torch.load(f'{dataset_root}/ssx_dataset_clean.pt')
res_dict = {}
for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    model_params = {
        'model_type': 'gat',
        'out_channels': 10,
        'num_layers': 2,
        'distmult': True,
    }

    run_hyperparams = {
        'seed': 42,
        'model_args': model_params,
        'num_iter': 5,
        'lr': 0.01,
        'epochs': 1000,
        'print_every': 10,
        'add_deg_feats': False,
        'batch_size': batch_size,
        'include_feats': ['integration2kmrank', 'integration10kmrank'],
    }
    res, metrics = run_all(dataset, run_args=run_hyperparams)
    res_dict[batch_size] = data
    torch.save(res_dict, f'{dataset_root}/model_runs/clean_batch_size_runs.pt')


        {'include_feats': ['x', 'y'] + integration_fields, 'scaler': 'quantile', 'feat_scaler_dict': {'x': 'minmax', 'y': 'minmax'}},
['degree'],
km2_fields,
km10_fields,
km100_fields,
choice_fields,
integration_fields,
nodecount_fields,
['choice2kmrank', 'choice10kmrank'],
['integration2kmrank', 'integration10kmrank'],
rank_fields,
unnorm_feature_fields,
og_feature_fields,
"""