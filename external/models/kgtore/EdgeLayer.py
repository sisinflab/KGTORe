import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor


class LGConv(MessagePassing):
    def __init__(self, alpha, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.alpha = alpha
        self.normalize = normalize

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, edge_attr_weight: OptTensor = None) -> Tensor:

        if self.normalize and isinstance(edge_index, Tensor):
            out = gcn_norm(edge_index, None, x.size(self.node_dim),
                           add_self_loops=False, flow=self.flow, dtype=x.dtype)
            edge_index, edge_weight = out
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr, edge_attr_weight=edge_attr_weight,
                              size=None)

    def message(self, x_j: Tensor, edge_weight, edge_attr: OptTensor, edge_attr_weight: OptTensor) -> Tensor:
        num_trans = x_j.shape[0] // 2
        x_j[num_trans:] = x_j[num_trans:] * self.alpha
        # e_a = torch.zeros(x_j.shape[0], x_j.shape[1])  # edge_attr for items
        # e_a[num_trans:] = edge_attr  # only for item
        # print(f" \n e_a_device:{e_a.device} \t x_j_device: {x_j.device} \t edge_weight: {edge_weight.device}")
        # return (edge_attr_weight.reshape(-1, 1) * edge_attr) + torch.mul(x_j, edge_weight.reshape(-1, 1))
        return edge_attr + torch.mul(x_j, edge_weight.reshape(-1, 1))

