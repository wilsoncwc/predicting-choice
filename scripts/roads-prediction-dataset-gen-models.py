#!/usr/bin/env python
# coding: utf-8

import torch
from ..train import run_single
from ..constants import dataset_root, rank_fields

def main():
    dataset = torch.load(f'{dataset_root}/ssx_dataset.pt')
    places = [data.place for data in dataset]
    half = len(places) // 2
    places = places[half:]
    
    model_params = {
        'model_type': 'gat',
        'distmult': True,
        'out_channels': 10
    }
    run_hyperparams = {
        'seed': 42,
        'model_args': model_params,
        'num_iter': 3,
        'lr': 0.01,
        'epochs': 300,
        'add_deg_feats': False,
        'include_feats': rank_fields,
        'save_best_model': False
    }
    run_single(places, dataset, run_args=run_hyperparams,
               save_path=f'{dataset_root}/link_pred_metrics_GATdistmult_iter2.pt')

if __name__ == '__main__':
    main()

