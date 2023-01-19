from abc import ABC

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
from torch.nn import Linear, Parameter
import torch


class EdgeLayer(MessagePassing, ABC):
    def __init__(self, alpha, beta, normalize=True):
        super(EdgeLayer, self).__init__(aggr='add')
        self.normalize = normalize
        self.alpha = alpha
        self.beta = beta
        # self.lin = Linear(64, 64, bias=False)
        # self.bias = Parameter(torch.nn.init.xavier_normal_(torch.empty(64)))
        # self.bias = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(1, 64)))

    def forward(self, x, edge_index, edge_attr):

        # x = self.lin(x)
        if self.normalize:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)# + self.bias
        else:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)# + self.bias

    def message(self, x_j, edge_attr):
        num_trans = x_j.shape[0] // 2
        x_j[:num_trans] = x_j[:num_trans] * self.beta
        x_j[num_trans:] = x_j[num_trans:] * self.alpha
        return edge_attr + x_j
