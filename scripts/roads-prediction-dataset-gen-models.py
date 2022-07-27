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
    dataset = torch.load(f'{dataset_root}/ssx_dataset_min.pt')
    places = [data.place for data in dataset]
    half = len(places) // 2
    places = places[half:]
    
    model_params = {
        'model_type': 'gain',
        'distmult': True,
        'out_channels': 10
    }
    data_process_args = {
        'include_feats': ['x', 'y'],
    }
    run_hyperparams = {
        'seed': 42,
        'model_args': model_params,
        'num_iter': 5,
        'lr': 0.01,
        'epochs': 500,
        'print_every': 10,
        'schedule_lr': True,
    }
    run_single(places, dataset, run_args=run_hyperparams, 
               data_process_args=data_process_args, only_transductive=False,
               save_path=f'{dataset_root}/link_pred/quantile_norm2.pt')

if __name__ == '__main__':
    main()

