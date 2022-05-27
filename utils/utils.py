import pandas as pd
import numpy as np
from numbers import Number

def remove_item(xs, ys):
    """List subtraction/Element removal without side effects"""
    if type(ys) != list:
        ys = [ys]
    return [item for item in xs if item not in ys]

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

def convert_categorical_features_to_one_hot(df, cat_fields):
    fields = df[cat_fields]
    one_hot_enc = pd.get_dummies(fields)
    df.drop(columns=fields)
    return pd.concat([df, one_hot_enc], axis=1), list(one_hot_enc.columns)