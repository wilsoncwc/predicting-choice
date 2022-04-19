#!/usr/bin/env python
# coding: utf-8

# Set path to include project root so that modules can be directly imported
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from train import run
from utils.constants import dataset_root

def main():
    dataset = torch.load(f'{dataset_root}/ssx_dataset_connected.pt')
    res_dict = {}
    for model_type in ['gcn', 'gat', 'sage', 'pna', 'gin', 'gain']:
        model_args = {
            'out_channels': 10,
            'model_type': model_type,
            'num_layers': 2,
            'distmult': False
        }
        proc_args = {
            'include_feats': ['integration2kmrank', 'integration10kmrank'],
            'add_deg_feats': False
        }
        print(f'Testing {model_type}')
        models, results = run(dataset,
                            proc_args,
                            model_args,
                            num_iter=5,
                            lr=0.01,
                            epochs=500,
                            schedule_lr=False,
                            output_tb=True,
                            save_best_model=False)
        res_dict[model_type] = results
        torch.save(res_dict, f'{dataset_root}/non-distmult-models-runs.pt')
    

if __name__ == '__main__':
    main()