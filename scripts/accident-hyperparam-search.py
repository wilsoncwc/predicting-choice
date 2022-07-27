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
from utils.constants import *
from utils.utils import remove_item

def main():
    print(f'Running training on GPU: {torch.cuda.get_device_name(0)}')
    place = remove_item(included_places, inductive_places)
    result_dict = {}
    filepath = f'{dataset_root}/accident_agg_inductive_15_runs2.pt'
    for var_args in [
        'max',
        'min',
        'sum',
        'mean',
        'median',
        'std'
    ]:
            data_process_args = {
                'split_approach': 'cluster',
                'num_parts': 512,
                'batch_size': 16,
                'include_feats': og_feature_fields + ['meridian_class'],
                'categorize': 'multiclass',
                'clean': True,
                'dist': 15,
                'agg': var_args
            }
            model_args = {
                'model_type': 'sage',
                'aggr': 'max',
                'num_layers': 2,
                'hidden_channels': 20,
            }
            print(f'Testing {var_args}')
            models, results = run(place,
                                  inductive_place=inductive_places,
                                  target_field='accident_count',
                                  data_process_args=data_process_args,
                                  model_args=model_args,
                                  lr=0.01,
                                  schedule_lr=True,
                                  criteria_names=['Accuracy', 'MAE', 'MSE', 'RMSE'],
                                  num_iter=5)
            result_dict[str(var_args)] = results
            torch.save(result_dict, filepath)
            print(f'Saving to {filepath}')
    

if __name__ == '__main__':
    main()

    
    
"""
for agg in [
        'max',
        'min',
        'sum',
        'mean',
        'median',
        'std'
    ]:

km2_fields,
km10_fields,
km100_fields,
choice_fields,
integration_fields,
nodecount_fields,
rank_fields,
og_feature_fields,
['degree'],
og_feature_fields + ['degree'],
geom_feats,
dual_feats,
['meridian_class'],
og_feature_fields + ['meridian_class']

{'model_type': 'mlp', 'num_layers': 2},
{'model_type': 'mlp', 'num_layers': 3},
{'model_type': 'mlp', 'num_layers': 4},
{'model_type': 'gcn', 'num_layers': 2},
{'model_type': 'gat', 'num_layers': 2},
{'model_type': 'sage', 'num_layers': 2, 'aggr': 'min'},
{'model_type': 'sage', 'num_layers': 2, 'aggr': 'mean'},
{'model_type': 'sage', 'num_layers': 2, 'aggr': 'max'},
{'model_type': 'sage', 'num_layers': 2, 'aggr': 'add'},
{'model_type': 'gin',  'num_layers': 2},
{'model_type': 'gain', 'num_layers': 2},

"""