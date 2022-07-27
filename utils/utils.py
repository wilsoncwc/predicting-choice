import pandas as pd
import numpy as np
import torch
from numbers import Number
from IPython.display import display

from utils.constants import known_cat_fields
def remove_item(xs, ys):
    """List subtraction/Element removal without side effects"""
    if type(ys) != list:
        ys = [ys]
    return [item for item in xs if item not in ys]

def add_item(xs, ys):
    if type(ys) != list:
        ys = [ys]
    for y in ys:
        if y not in xs:
            xs = xs + [y]
    return xs

def apply_agg(xs, agg):
    if isinstance(xs[0], Number): 
        if agg == 'sum':
            return sum(xs)
        if agg == 'max':
            return max(xs)
        if agg == 'min':
            return min(xs)
        if agg == 'mean':
            return np.mean(xs, axis=0)
        if agg == 'std':
            return np.std(xs, axis=0)
        if agg == 'median':
            return np.median(xs, axis=0)
        else:
            raise ValueError
    else: # Mode for non-numeric values
        return max(set(xs), key=xs.count)

def mean_curves(curves):
    """Get the average of differently sized curves with max value 1"""
    max_len = max(len(curve) for curve in curves)
    padded_curves = []
    for curve in curves:
        diff = max_len - len(curve)
        padded_curve = np.pad(curve, (0, diff), mode='constant', constant_values=1)
        padded_curves.append(padded_curve)
    return np.mean(padded_curves, axis=0)


def macro_mae(y_true, y_pred):
    # Round predictions to the nearest integer
    return macro_averaged_mean_absolute_error(y_true, torch.round(y_pred))

def summarize_results(results, average=False):
    """Summarize run result dict returned by train.run"""
    # Obtain final metrics of each run
    data = {}
    for result in results:
        for key in result:
            metric_to_append = None
            if type(result[key]) == list:
                if len(result[key]) > 0:
                    metric_to_append = result[key][-1]
            else:
                metric_to_append = result[key]
            if metric_to_append is not None:
                if key in data:
                    data[key].append(metric_to_append)
                else:
                    data[key] = [metric_to_append]
    
    if average:
        # Compute average over run (if metric is numeric), or take the first element
        for key in data:
            if len(data[key]) > 0:
                if isinstance(data[key][0], Number):
                    data[key] = sum(data[key]) / len(data[key])
                else:
                    data[key] = data[key][0]
    
    return data

def display_results(results):
    result_list = []
    for result in results:
        d = {
            key: result[key][-1] \
                if type(result[key]) == list and len(result[key]) > 0 \
                else result[key] \
            for key in result
        }
        result_list.append(d)
    result_df = pd.DataFrame.from_records(result_list)
    display(result_df)
    
def filter_feats(data, include_feats):
    feat_idx = torch.tensor([data.node_attrs.index(feat) for feat in include_feats])
    feat_idx = feat_idx.to(data.x.device)
    data.num_nodes = data.x.size(0)
    data.x = torch.index_select(data.x, 1, feat_idx).float() \
            if len(feat_idx) > 0 else None
    data.node_attrs = [feat for feat in data.node_attrs if feat in include_feats]

def convert_categorical_features_to_one_hot(g, cat_fields):
    field_list = {field: [] for field in cat_fields}
    for _, d in g.nodes(data=True):
        for field in cat_fields:
            field_list[field].append(d[field])
    df = pd.DataFrame.from_records(field_list)
    new_cols = []
    for field in cat_fields:
        one_hot_enc = pd.get_dummies(df[field])
        
        # Appending missing categories if known
        if field in known_cat_fields:
            known_cols = known_cat_fields[field]
            for col in known_cols:
                if col not in one_hot_enc.columns:
                    one_hot_enc[col] = 0
            # Reorder
            one_hot_enc = one_hot_enc[known_cols]
            
        cols = list(one_hot_enc.columns)
        new_cols += [f'{field}_{col}' for col in cols]
        for idx, (_, d) in enumerate(g.nodes(data=True)):
            del d[field]
            for col in cols:
                d[f'{field}_{col}'] = one_hot_enc.iloc[idx][col]
    return g, new_cols