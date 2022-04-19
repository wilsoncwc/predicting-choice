import torch
from torch.nn import Linear, ModuleList
from torch_geometric.nn import MLP, GCNConv, GATConv, SAGEConv, GraphConv, global_sort_pool
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, PNA
from torch_geometric.nn import GAE, VGAE, InnerProductDecoder

from sklearn.metrics import roc_curve, precision_recall_curve

from gain import GAIN

def init_conv(layer_type):
    if layer_type == 'gcn':
        return GCNConv
    if layer_type == 'gat':
        return GATConv
    if layer_type == 'sage':
        return SAGEConv
    if layer_type == 'graph':
        return GraphConv

def init_conv_model(model_type, **kwargs):
    if model_type == 'gcn':
        return GCN(**kwargs)
    if model_type == 'gat':
        return GAT(**kwargs)
    if model_type == 'sage':
        return GraphSAGE(**kwargs)
    if model_type == 'gin':
        return GIN(**kwargs)
    if model_type == 'gain':
        return GAIN(**kwargs)
    if model_type == 'pna':
        return PNA(**kwargs)
        
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, model_type='gcn', **kwargs):
        super().__init__()
        self.conv = init_conv_model(model_type, 
                                    in_channels=in_channels, 
                                    out_channels=out_channels, 
                                    hidden_channels=out_channels,
                                    num_layers=num_layers,
                                    **kwargs)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, model_type='gcn', **kwargs):
        super().__init__()
        self.conv = init_conv_model(model_type, 
                                    in_channels=in_channels, 
                                    out_channels=out_channels * 2, 
                                    hidden_channels=out_channels,
                                    num_layers=num_layers,
                                    **kwargs)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        return torch.chunk(h, 2)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class DistMultDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, z, edge_index, sigmoid=True):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        int_val = torch.matmul(self.weights, z_dst.t())
        value = torch.sum(z_src * int_val.t(), dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, torch.matmul(self.weights, z.t()))
        return torch.sigmoid(adj) if sigmoid else adj
    
    
class Mod():
    def predict_pos(self, data, threshold=0.6):
        z = self.encode(data.x, data.edge_index)
        logits = self.decode(z, data.edge_index).view(-1)
        pos_logits = self.decode(z, data.pos_edge_label_index).view(-1)
        logits = torch.cat([logits, pos_logits]).detach().cpu()
        return (logits > threshold).float(), logits

    # Requires negative sampled data (RandomLinkSplit with split_label=True)
    def predict_neg(self, data, threshold=0.6, enhanced=True):
        z = self.encode(data.x, data.edge_index)
        logits = self.decode(z, data.neg_edge_label_index).view(-1)
        if not enhanced:
            return (logits > threshold).float().cpu(), logits
        node_coords = torch.stack((data.lng, data.lat), dim=-1).detach().cpu()
        preds = []
        neg_edges = torch.stack((data.neg_edge_label_index[0], data.neg_edge_label_index[1]), dim=-1)
        for idx, label_idx in enumerate(neg_edges):
            logit = logits[idx]
            if logit > threshold:
                if is_valid(label_idx, node_coords, data.edge_index):
                    preds.append(1)
                else:
                    preds.append(0)
            else:
                preds.append(0)
        return torch.tensor(preds), logits
    
    def test_curves(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_curve(y, pred), precision_recall_curve(y, pred)

    
class ModGAE(GAE, Mod):
    pass


class ModVGAE(VGAE, Mod):
    pass


def init_model(linear=False, variational=False, distmult=True, **model_args):
    decoder = InnerProductDecoder() if not distmult else DistMultDecoder(model_args['out_channels'])
    if not variational and not linear:
        model = ModGAE(GCNEncoder(**model_args), decoder)
    elif not variational and linear:
        model = ModGAE(LinearEncoder(**model_args), decoder)
    elif variational and not linear:
        model = ModVGAE(VariationalGCNEncoder(**model_args), decoder)
    elif variational and linear:
        model = ModVGAE(VariationalLinearEncoder(**model_args), decoder)
    print(f'Initialized {model} with arguments {model_args}')
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model

