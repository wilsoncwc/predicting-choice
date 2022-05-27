#!/usr/bin/env python
# coding: utf-8

# Set path to include project root so that modules can be directly imported
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from prediction import run
from utils.constants import dataset_root, rank_fields, log_fields, all_feature_fields, geom_feats, unnorm_feature_fields, dual_feats
def remove_item(xs, ys):
    if type(ys) != list:
        ys = [ys]
    return [item for item in xs if item not in ys]

def main():
    place = 'No Bounds'
    result_dict = {}
    filepath = f'{dataset_root}/accident_reg_pred_model_runs.pt'
    for model_args in [
        {'model_type': 'gcn', 'hidden_channels': 10, 'num_layers': 2},
        {'model_type': 'gat', 'hidden_channels': 10, 'num_layers': 2},
        {'model_type': 'sage', 'hidden_channels': 10, 'num_layers': 2},
        {'model_type': 'gin', 'hidden_channels': 10, 'num_layers': 2},
        {'model_type': 'gain', 'hidden_channels': 10, 'num_layers': 2},
    ]:
            data_process_args = {
                'split_approach': 'cluster',
                'include_feats': all_feature_fields,
                'categorize': 'regression',
                'clean': True
            }
            print(f'Testing {model_args}')
            models, results = run(place, target_field='accident_count',
                                  data_process_args=data_process_args,
                                  model_args=model_args,
                                  lr=0.005,
                                  num_iter=5)
            result_dict[str(model_args)] = results
            torch.save(result_dict, filepath)
            print(f'Saving to {filepath}')
    

if __name__ == '__main__':
    main()
    