import torch


class Aggregator(torch.nn.Module):
    """GNN Aggregator layer"""

    def __init__(self, input_dim, output_dim, dropout, aggregator_type, device):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.device = device

        self.message_dropout = torch.nn.Dropout(dropout)

        if self.aggregator_type == "gcn":
            self.W = torch.nn.Linear(self.input_dim, self.output_dim)
            self.W.to(self.device)
        elif self.aggregator_type == "graphsage":
            self.W = torch.nn.Linear(self.input_dim * 2, self.output_dim)
            self.W.to(self.device)
        elif self.aggregator_type == "bi":
            self.W1 = torch.nn.Linear(self.input_dim, self.output_dim)
            self.W2 = torch.nn.Linear(self.input_dim, self.output_dim)
            self.W1.to(self.device)
            self.W2.to(self.device)
        else:
            raise NotImplementedError

        self.activation = torch.nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == "gcn":
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == "graphsage":
            ego_embeddings = self.activation(
                self.W(torch.cat([ego_embeddings, side_embeddings], dim=1))
            )
        elif self.aggregator_type == "bi":
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings
