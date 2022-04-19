#!/usr/bin/env python
# coding: utf-8

# Set path to include project root so that modules can be directly imported
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from train import run_single
from utils.constants import dataset_root

def main():
    dataset = torch.load(f'{dataset_root}/ssx_dataset_connected.pt')
    places = [data.place for data in dataset]
    half = len(places) // 2
    # places = places[half:]
    places = ['Coventry']
    
    model_params = {
        'model_type': 'gat',
        'distmult': True,
        'out_channels': 10
    }
    run_hyperparams = {
        'seed': 42,
        'model_args': model_params,
        'num_iter': 1,
        'lr': 0.01,
        'epochs': 300,
        'add_deg_feats': False,
        'include_feats': ['integration2kmrank', 'integration10kmrank'],
        'save_best_model': False
    }
    run_single(places, dataset, run_args=run_hyperparams)
               # save_path=f'{dataset_root}/link_pred_metrics_GATdistmult_iter1.pt')

if __name__ == '__main__':
    main()

