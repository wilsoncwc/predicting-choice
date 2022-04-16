import torch
from torch.nn import Linear, ModuleList
from torch_geometric.nn import MLP, GCNConv, GATConv, SAGEConv, GraphConv, global_sort_pool
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT, PNA
from torch_geometric.nn import GAE, VGAE, InnerProductDecoder

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
    print(kwargs)
    if model_type == 'gcn':
        return GCN(**kwargs)
    if model_type == 'gat':
        return GAT(**kwargs)
    if model_type == 'sage':
        return GraphSAGE(**kwargs)
    if model_type == 'gin':
        return GIN(**kwargs)
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
    
    
class MyGAE(GAE):
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

def init_model(linear=False, variational=False, distmult=True, **model_args):
    decoder = InnerProductDecoder() if not distmult else DistMultDecoder(model_args['out_channels'])
    if not variational and not linear:
        model = MyGAE(GCNEncoder(**model_args), decoder)
    elif not variational and linear:
        model = MyGAE(LinearEncoder(**model_args), decoder)
    elif variational and not linear:
        model = VGAE(VariationalGCNEncoder(**model_args), decoder)
    elif variational and linear:
        model = VGAE(VariationalLinearEncoder(**model_args), decoder)
    return model

# from typing import Callable, Optional, Union

# import torch
# from torch import Tensor
# from torch_sparse import SparseTensor, matmul

# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

# from ..inits import reset

# class GAINConv(MessagePassing):
#     r"""The graph isomorphism operator from the `"How Powerful are
#     Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

#     .. math::
#         \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
#         \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

#     or

#     .. math::
#         \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
#         (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

#     here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

#     Args:
#         nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
#             maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
#             shape :obj:`[-1, out_channels]`, *e.g.*, defined by
#             :class:`torch.nn.Sequential`.
#         eps (float, optional): (Initial) :math:`\epsilon`-value.
#             (default: :obj:`0.`)
#         train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
#             will be a trainable parameter. (default: :obj:`False`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.

#     Shapes:
#         - **input:**
#           node features :math:`(|\mathcal{V}|, F_{in})` or
#           :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
#           if bipartite,
#           edge indices :math:`(2, |\mathcal{E}|)`
#         - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
#           :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
#     """
#     def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
#                  **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)
#         self.nn = nn
#         self.initial_eps = eps
#         if train_eps:
#             self.eps = torch.nn.Parameter(torch.Tensor([eps]))
#         else:
#             self.register_buffer('eps', torch.Tensor([eps]))
#         self.reset_parameters()

# [docs]
#     def reset_parameters(self):
#         reset(self.nn)
#         self.eps.data.fill_(self.initial_eps)


# [docs]
#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 size: Size = None) -> Tensor:
#         """"""
#         if isinstance(x, Tensor):
#             x: OptPairTensor = (x, x)

#         # propagate_type: (x: OptPairTensor)
#         out = self.propagate(edge_index, x=x, size=size)

#         x_r = x[1]
#         if x_r is not None:
#             out += (1 + self.eps) * x_r

#         return self.nn(out)


#     def message(self, x_j: Tensor) -> Tensor:
#         return x_j

#     def message_and_aggregate(self, adj_t: SparseTensor,
#                               x: OptPairTensor) -> Tensor:
#         adj_t = adj_t.set_value(None, layout=None)
#         return matmul(adj_t, x[0], reduce=self.aggr)

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(nn={self.nn})'
    
    
# class GATConv(MessagePassing):
#     r"""The graph attentional operator from the `"Graph Attention Networks"
#     <https://arxiv.org/abs/1710.10903>`_ paper

#     .. math::
#         \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
#         \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

#     where the attention coefficients :math:`\alpha_{i,j}` are computed as

#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
#         \right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
#         \right)\right)}.

#     If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
#     the attention coefficients :math:`\alpha_{i,j}` are computed as

#     .. math::
#         \alpha_{i,j} =
#         \frac{
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
#         \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
#         {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
#         \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
#         [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
#         \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

#     Args:
#         in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
#             derive the size from the first input(s) to the forward method.
#             A tuple corresponds to the sizes of source and target
#             dimensionalities.
#         out_channels (int): Size of each output sample.
#         heads (int, optional): Number of multi-head-attentions.
#             (default: :obj:`1`)
#         concat (bool, optional): If set to :obj:`False`, the multi-head
#             attentions are averaged instead of concatenated.
#             (default: :obj:`True`)
#         negative_slope (float, optional): LeakyReLU angle of the negative
#             slope. (default: :obj:`0.2`)
#         dropout (float, optional): Dropout probability of the normalized
#             attention coefficients which exposes each node to a stochastically
#             sampled neighborhood during training. (default: :obj:`0`)
#         add_self_loops (bool, optional): If set to :obj:`False`, will not add
#             self-loops to the input graph. (default: :obj:`True`)
#         edge_dim (int, optional): Edge feature dimensionality (in case
#             there are any). (default: :obj:`None`)
#         fill_value (float or Tensor or str, optional): The way to generate
#             edge features of self-loops (in case :obj:`edge_dim != None`).
#             If given as :obj:`float` or :class:`torch.Tensor`, edge features of
#             self-loops will be directly given by :obj:`fill_value`.
#             If given as :obj:`str`, edge features of self-loops are computed by
#             aggregating all features of edges that point to the specific node,
#             according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
#             :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.

#     Shapes:
#         - **input:**
#           node features :math:`(|\mathcal{V}|, F_{in})` or
#           :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
#           if bipartite,
#           edge indices :math:`(2, |\mathcal{E}|)`,
#           edge features :math:`(|\mathcal{E}|, D)` *(optional)*
#         - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
#           :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
#           If :obj:`return_attention_weights=True`, then
#           :math:`((|\mathcal{V}|, H * F_{out}),
#           ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
#           or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
#           (|\mathcal{E}|, H)))` if bipartite
#     """
#     def __init__(
#         self,
#         in_channels: Union[int, Tuple[int, int]],
#         out_channels: int,
#         heads: int = 1,
#         concat: bool = True,
#         negative_slope: float = 0.2,
#         dropout: float = 0.0,
#         add_self_loops: bool = True,
#         edge_dim: Optional[int] = None,
#         fill_value: Union[float, Tensor, str] = 'mean',
#         bias: bool = True,
#         **kwargs,
#     ):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(node_dim=0, **kwargs)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.concat = concat
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.add_self_loops = add_self_loops
#         self.edge_dim = edge_dim
#         self.fill_value = fill_value

#         # In case we are operating in bipartite graphs, we apply separate
#         # transformations 'lin_src' and 'lin_dst' to source and target nodes:
#         if isinstance(in_channels, int):
#             self.lin_src = Linear(in_channels, heads * out_channels,
#                                   bias=False, weight_initializer='glorot')
#             self.lin_dst = self.lin_src
#         else:
#             self.lin_src = Linear(in_channels[0], heads * out_channels, False,
#                                   weight_initializer='glorot')
#             self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
#                                   weight_initializer='glorot')

#         # The learnable parameters to compute attention coefficients:
#         self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
#         self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

#         if edge_dim is not None:
#             self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
#                                    weight_initializer='glorot')
#             self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
#         else:
#             self.lin_edge = None
#             self.register_parameter('att_edge', None)

#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

# [docs]
#     def reset_parameters(self):
#         self.lin_src.reset_parameters()
#         self.lin_dst.reset_parameters()
#         if self.lin_edge is not None:
#             self.lin_edge.reset_parameters()
#         glorot(self.att_src)
#         glorot(self.att_dst)
#         glorot(self.att_edge)
#         zeros(self.bias)


# [docs]
#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 edge_attr: OptTensor = None, size: Size = None,
#                 return_attention_weights=None):
#         # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
#         # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
#         # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
#         r"""
#         Args:
#             return_attention_weights (bool, optional): If set to :obj:`True`,
#                 will additionally return the tuple
#                 :obj:`(edge_index, attention_weights)`, holding the computed
#                 attention weights for each edge. (default: :obj:`None`)
#         """
#         # NOTE: attention weights will be returned whenever
#         # `return_attention_weights` is set to a value, regardless of its
#         # actual value (might be `True` or `False`). This is a current somewhat
#         # hacky workaround to allow for TorchScript support via the
#         # `torch.jit._overload` decorator, as we can only change the output
#         # arguments conditioned on type (`None` or `bool`), not based on its
#         # actual value.

#         H, C = self.heads, self.out_channels

#         # We first transform the input node features. If a tuple is passed, we
#         # transform source and target node features via separate weights:
#         if isinstance(x, Tensor):
#             assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
#             x_src = x_dst = self.lin_src(x).view(-1, H, C)
#         else:  # Tuple of source and target node features:
#             x_src, x_dst = x
#             assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
#             x_src = self.lin_src(x_src).view(-1, H, C)
#             if x_dst is not None:
#                 x_dst = self.lin_dst(x_dst).view(-1, H, C)

#         x = (x_src, x_dst)

#         # Next, we compute node-level attention coefficients, both for source
#         # and target nodes (if present):
#         alpha_src = (x_src * self.att_src).sum(dim=-1)
#         alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
#         alpha = (alpha_src, alpha_dst)

#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 # We only want to add self-loops for nodes that appear both as
#                 # source and target nodes:
#                 num_nodes = x_src.size(0)
#                 if x_dst is not None:
#                     num_nodes = min(num_nodes, x_dst.size(0))
#                 num_nodes = min(size) if size is not None else num_nodes
#                 edge_index, edge_attr = remove_self_loops(
#                     edge_index, edge_attr)
#                 edge_index, edge_attr = add_self_loops(
#                     edge_index, edge_attr, fill_value=self.fill_value,
#                     num_nodes=num_nodes)
#             elif isinstance(edge_index, SparseTensor):
#                 if self.edge_dim is None:
#                     edge_index = set_diag(edge_index)
#                 else:
#                     raise NotImplementedError(
#                         "The usage of 'edge_attr' and 'add_self_loops' "
#                         "simultaneously is currently not yet supported for "
#                         "'edge_index' in a 'SparseTensor' form")

#         # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
#         alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

#         # propagate_type: (x: OptPairTensor, alpha: Tensor)
#         out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

#         if self.concat:
#             out = out.view(-1, self.heads * self.out_channels)
#         else:
#             out = out.mean(dim=1)

#         if self.bias is not None:
#             out += self.bias

#         if isinstance(return_attention_weights, bool):
#             if isinstance(edge_index, Tensor):
#                 return out, (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out


# [docs]
#     def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
#                     edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
#                     size_i: Optional[int]) -> Tensor:
#         # Given edge-level attention coefficients for source and target nodes,
#         # we simply need to sum them up to "emulate" concatenation:
#         alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

#         if edge_attr is not None:
#             if edge_attr.dim() == 1:
#                 edge_attr = edge_attr.view(-1, 1)
#             assert self.lin_edge is not None
#             edge_attr = self.lin_edge(edge_attr)
#             edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
#             alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
#             alpha = alpha + alpha_edge

#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, index, ptr, size_i)
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         return alpha


#     def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
#         return alpha.unsqueeze(-1) * x_j

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, heads={self.heads})')