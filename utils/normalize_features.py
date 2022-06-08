from typing import List, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from sklearn.preprocessing import robust_scale, maxabs_scale, minmax_scale, power_transform

def sum_to_one_scale(X):
    X = X - X.min()
    return X.div_(X.sum(dim=-1, keepdim=True).clamp_(min=1.))

scaler_dict = {
    'robust': robust_scale,
    'maxabs': maxabs_scale,
    'minmax': minmax_scale,
    'power': power_transform,
    'sum': sum_to_one_scale
}

class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, attrs: List[str] = ["x"], scaler='sum'):
        self.attrs = attrs
        self.scaler = scaler_dict[scaler]

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                store[key] = torch.tensor(self.scaler(value), dtype=value.dtype)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'