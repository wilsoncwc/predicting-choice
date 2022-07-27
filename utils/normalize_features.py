from typing import List, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from functools import partial
from sklearn.preprocessing import robust_scale, maxabs_scale, minmax_scale, power_transform, quantile_transform

def sum_to_one_scale(X):
    X = X - X.min()
    return X.div_(X.sum(dim=0, keepdim=True).clamp_(min=1.))

scaler_dict = {
    'none': (lambda x: x),
    'robust': robust_scale,
    'maxabs': maxabs_scale,
    'minmax': minmax_scale,
    'power': power_transform,
    'sum': sum_to_one_scale,
    'quantile': quantile_transform,
    'quantile-normal': partial(quantile_transform, output_distribution='normal')
}

class NormalizeFeatures(BaseTransform):
    r"""Normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
        scaler (str): The normalization operator
            (default: "sum")
    """
    def __init__(self, attrs: List[str] = ["x"], scaler='sum', feat_scaler_dict={}):
        self.attrs = attrs
        self.scaler = scaler_dict[scaler]
        self.scaler_dict = feat_scaler_dict

    def __call__(self, data: Union[Data, HeteroData]):
        # new_x = []
        # for idx, attr in enumerate(data.node_attrs):
        #     if attr not in self.scaler_dict:
        #         self.scaler_dict[attr] = self.scaler
        #     else:
        #         self.scaler_dict[attr] = scaler_dict[self.scaler_dict[attr]]
        #     new_x.append(scaler(data.x[:, idx:idx+1]))
        # new_x = torch.cat(new_x, dim=1)
        
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if key == 'x':
                    new_x = []
                    for idx, attr in enumerate(data.node_attrs):
                        scaler = self.scaler \
                            if attr not in self.scaler_dict \
                            else scaler_dict[self.scaler_dict[attr]]
                        scaled = scaler(value[:, idx:idx+1])
                        new_x.append(torch.tensor(scaled, dtype=value.dtype))

                    scaled_x = torch.cat(new_x, dim=1)
        
#                     trunc_value = torch.index_select(value, 1, torch.tensor(incl_idx))
                    
#                     scaled_x = torch.tensor(self.scaler(trunc_value), dtype=value.dtype)
#                     for idx in excl_idx:
#                         scaled_x = torch.cat((scaled_x[:, :idx], value[:, idx:idx+1], scaled_x[:, idx:]), dim=1)
                else:
                    scaled_x = torch.tensor(self.scaler(value), dtype=value.dtype)
                    
                store[key] = scaled_x
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'