import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor


class LGConv(MessagePassing):
    r"""The Light Graph Convolution (LGC) operator from the `"LightGCN:
    Simplifying and Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{e_{j,i}}{\sqrt{\deg(i)\deg(j)}} \mathbf{x}_j

    Args:
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be normalized via symmetric normalization.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    def __init__(self, alpha, beta, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.normalize = normalize

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, edge_attr_weight: OptTensor = None) -> Tensor:

        if self.normalize and isinstance(edge_index, Tensor):
            out = gcn_norm(edge_index, None, x.size(self.node_dim),
                           add_self_loops=False, flow=self.flow, dtype=x.dtype)
            edge_index, edge_weight = out
        elif self.normalize and isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim),
                                  add_self_loops=False, flow=self.flow,
                                  dtype=x.dtype)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr, edge_attr_weight=edge_attr_weight,
                              size=None)

    def message(self, x_j: Tensor, edge_weight, edge_attr: OptTensor, edge_attr_weight: OptTensor) -> Tensor:
        num_trans = x_j.shape[0] // 2
        x_j[:num_trans] = x_j[:num_trans] * self.beta
        x_j[num_trans:] = x_j[num_trans:] * self.alpha
        return (edge_attr_weight.reshape(-1, 1) * edge_attr) + torch.mul(x_j, edge_weight.reshape(-1, 1))

