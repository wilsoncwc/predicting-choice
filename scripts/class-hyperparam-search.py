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
    filepath = f'{project_root}/meridian_inductive_model_runs.pt'
    for var_args in  [
        {'model_type': 'mlp', 'num_layers': 2},
        {'model_type': 'mlp', 'num_layers': 3},
        {'model_type': 'mlp', 'num_layers': 4},
        {'model_type': 'gcn', 'num_layers': 2},
        {'model_type': 'gat', 'num_layers': 2},
        {'model_type': 'sage', 'num_layers': 2, 'aggr': 'min'},
        {'model_type': 'sage', 'num_layers': 2, 'aggr': 'mean'},
        {'model_type': 'sage', 'num_layers': 2, 'aggr': 'add'},
        {'model_type': 'gin',  'num_layers': 2},
        {'model_type': 'gain', 'num_layers': 2},
    ]:
            data_process_args = {
                'split_approach': 'neighbor',
                'batch_size': 4096,
                'include_feats': unnorm_feature_fields,
                'clean': False,
                # **var_args
            }
            model_args = {
                **var_args,
                'hidden_channels': 20
            }
            print(f'Testing {var_args}')
            models, results = run(place,
                                  inductive_place=inductive_places,
                                  target_field='meridian_class',
                                  data_process_args=data_process_args,
                                  model_args=model_args,
                                  schedule_lr=True,
                                  lr=0.01,
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
    
{ 'split_approach': 'cluster', 'num_parts': 512, 'batch_size': 2 },
{ 'split_approach': 'cluster', 'num_parts': 512, 'batch_size': 4 },
{ 'split_approach': 'cluster', 'num_parts': 512, 'batch_size': 8 },
{ 'split_approach': 'cluster', 'num_parts': 1024, 'batch_size': 1 },
{ 'split_approach': 'cluster', 'num_parts': 1024, 'batch_size': 2 },
{ 'split_approach': 'cluster', 'num_parts': 1024, 'batch_size': 4 },
{ 'split_approach': 'cluster', 'num_parts': 1024, 'batch_size': 8 },
    
{ 'split_approach': 'neighbor', 'batch_size': 100 },
{ 'split_approach': 'neighbor', 'batch_size': 1000 },
{ 'split_approach': 'neighbor', 'batch_size': 10000 },
{ 'split_approach': 'saint', 'batch_size': 100 },
{ 'split_approach': 'saint', 'batch_size': 1000 },
{ 'split_approach': 'saint', 'batch_size': 10000 },
{ 'split_approach': 'none' }

"""
