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
from utils.constants import dataset_root

def main():
    result_dict = {}
    for var in [
        'ssx_dataset_max',
        'ssx_dataset_min',
        'ssx_dataset_sum',
        'ssx_dataset_mean',
        'ssx_dataset_median',
        'ssx_dataset_std'
    ]:
        dataset = torch.load(f'{dataset_root}/{var}.pt')
        model_args = {
            'out_channels': 10,
            'model_type': 'gain',
            'num_layers': 2,
            'distmult': True,
        }
        proc_args = {
            'include_feats': ['integration2kmrank', 'integration10kmrank'],
            'add_deg_feats': False
        }
        print(f'Testing {var}...')
        models, results = run(dataset,
                            proc_args,
                            model_args,
                            num_iter=5,
                            lr=0.001,
                            epochs=500,
                            schedule_lr=False,
                            output_tb=False,
                            save_best_model=False)
        result_dict[var] = results
        torch.save(result_dict, f'{dataset_root}/model_runs/raw_agg_runs.pt')

if __name__ == '__main__':
    main()
    
    
    
"""
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
"""