from abc import ABC

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
from torch.nn import Linear, Parameter
import torch


class EdgeLayer(MessagePassing, ABC):
    def __init__(self, normalize=True):
        super(EdgeLayer, self).__init__(aggr='add')
        self.normalize = normalize
        # self.lin = Linear(64, 64, bias=False)
        # self.bias = Parameter(torch.Tensor(64))


    def forward(self, x, edge_index, edge_attr):

        # x = self.lin(x)

        if self.normalize:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr) #  + self.bias
        else:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr) # + self.bias

    def message(self, x_j, edge_attr):
        # return torch.unsqueeze(torch.mean(edge_attr, dim=-1), dim=-1) * x_j
        return edge_attr * x_j
