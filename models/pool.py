import torch

from torch_geometric.nn import Linear, global_add_pool, global_mean_pool, global_max_pool, global_sort_pool
from models.init_model import init_gnn_model

def init_pool(pool_type):
    if pool_type == 'add':
        return global_add_pool
    if pool_type == 'max':
        return global_max_pool
    if pool_type == 'mean':
        return global_mean_pool
    if pool_type == 'sort':
        return global_sort_pool
        
class PoolGNN(torch.nn.Module):
    def __init__(self, model_type='gat', pool_type='add', model_kwargs={}, pool_kwargs={}):
        super().__init__()
        self.pool = init_pool(pool_type)
        self.lin = Linear(model_kwargs['hidden_channels'], model_kwargs['out_channels'])
        
        model_kwargs_ = {**model_kwargs}
        model_kwargs_['out_channels'] = model_kwargs_['hidden_channels']
        self.conv = init_gnn_model(model_type, **model_kwargs_)
        self.pool_kwargs = pool_kwargs

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        x = self.pool(x, batch, **self.pool_kwargs)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x