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
    filepath = f'{dataset_root}/meridian_class_pred_loader_runs.pt'
    for var_args in  [
        { 'split_approach': 'cluster', 'batch_size': 8 },
        { 'split_approach': 'cluster', 'batch_size': 16 },
        { 'split_approach': 'cluster', 'batch_size': 32 },
        { 'split_approach': 'cluster', 'batch_size': 64 },
        { 'split_approach': 'cluster', 'batch_size': 128 },
        { 'split_approach': 'cluster', 'batch_size': 256 },
        { 'split_approach': 'cluster', 'batch_size': 512 },
    ]:
            data_process_args = {
                'split_approach': 'cluster',
                'include_feats': all_feature_fields,
                'add_deg_feats': True,
                'clean': False,
                **var_args
            }
            model_args = {
                'model_type': 'gain',
                'hidden_channels': 20,
                'num_layers': 2,
            }
            print(f'Testing {model_args}')
            models, results = run(place, target_field='meridian_class',
                                  data_process_args=data_process_args,
                                  model_args=model_args,
                                  lr=0.005,
                                  num_iter=5)
            result_dict[str(var_args)] = results
            torch.save(result_dict, filepath)
            print(f'Saving to {filepath}')
    

if __name__ == '__main__':
    main()
    
"""
class_pred_feat_runs1.pt
['choice2km', 'nodecount2km', 'integration2km'],
['choice10km', 'nodecount10km', 'integration10km'],
['choice100km', 'nodecount100km', 'integration100km'],
['choice2km', 'choice10km', 'choice100km'],
['integration2km', 'integration10km', 'integration100km'],
['nodecount2km', 'nodecount10km', 'nodecount100km'],
unnorm_feature_fields

class_pred_feat_runs2.pt
['degree'],
rank_fields,
rank_fields + ['degree'],
all_feature_fields,
all_feature_fields + ['degree'],
geom_feats,
dual_feats

accident_class_pred_feat_runs.pt
['degree'],
        geom_feats,
        ['choice2km', 'nodecount2km', 'integration2km'],
        ['choice10km', 'nodecount10km', 'integration10km'],
        ['choice100km', 'nodecount100km', 'integration100km'],
        ['choice2km', 'choice10km', 'choice100km'],
        ['integration2km', 'integration10km', 'integration100km'],
        ['nodecount2km', 'nodecount10km', 'nodecount100km'],
        rank_fields,
        log_fields,
        unnorm_feature_fields,
        all_feature_fields,
        
meridian_class_pred_feat_runs_raw
['degree'],
geom_feats,
['choice2km', 'nodecount2km', 'integration2km'],
['choice10km', 'nodecount10km', 'integration10km'],
['choice100km', 'nodecount100km', 'integration100km'],
['choice2km', 'choice10km', 'choice100km'],
['integration2km', 'integration10km', 'integration100km'],
['nodecount2km', 'nodecount10km', 'nodecount100km'],
rank_fields,
log_fields,
geom_feats,
unnorm_feature_fields,
all_feature_fields,
all_feature_fields + ['degree'],
dual_feats


 [
        {'model_type': 'mlp', 'hidden_channels': 10, 'num_layers': 2},
        {'model_type': 'mlp', 'hidden_channels': 10, 'num_layers': 3},
        {'model_type': 'mlp', 'hidden_channels': 10, 'num_layers': 4},
        {'model_type': 'gcn', 'hidden_channels': 10, 'num_layers': 2},
        {'model_type': 'gat', 'hidden_channels': 10, 'num_layers': 2},
        {'model_type': 'sage', 'hidden_channels': 10, 'num_layers': 2},
        {'model_type': 'gin', 'hidden_channels': 10, 'num_layers': 2},
        {'model_type': 'gain', 'hidden_channels': 10, 'num_layers': 2},
    ]
"""
