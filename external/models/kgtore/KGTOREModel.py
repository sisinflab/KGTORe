from abc import ABC

from .EdgeLayer import LGConv
import torch
import torch_geometric
import numpy as np
import random
from torch_sparse import matmul


class KGTOREModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_interactions,
                 learning_rate,
                 edges_lr,
                 embedding_size,
                 l_w,
                 alpha,
                 beta,
                 gamma,
                 n_layers,
                 edge_index,
                 edge_features,
                 item_features,
                 random_seed,
                 name="KGTORE",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        self.num_users = num_users
        self.num_items = num_items
        self.num_interactions = num_interactions
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.edges_lr = edges_lr
        self.l_w = l_w
        self.gamma = gamma
        self.n_layers = n_layers
        self.weight_size_list = [self.embedding_size] * (self.n_layers + 1)
        self.alpha = torch.tensor([1 / (k + 1) for k in range(len(self.weight_size_list))])
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64, device=self.device)
        self.edge_features = edge_features
        self.edge_features.to(self.device)
        self.item_features = item_features
        self.item_features.to(self.device)

        _, self.cols = self.edge_index.clone()
        self.items = self.cols[:self.num_interactions]
        self.items -= self.num_users

        # ADDITIVE OPTIONS
        self.a = alpha
        self.b = beta

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embedding_size))).to(self.device), requires_grad=True)
        #self.Gu.requires_grad_(True)
        print(self.Gu.get_device())

        #self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embedding_size))))
        self.Gi.to(self.device)

        # features matrix (for edges)
        self.feature_dim = edge_features.size(1)
        self.F = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.feature_dim, self.embedding_size)))
        ).to(self.device)
        print('primo device')
        print(self.device)
        print('primo device')
        print(self.device)
        print(self.edge_features.device())
        print(self.F.get_device())

        propagation_network_list = []

        for layer in range(self.n_layers):
            propagation_network_list.append((LGConv(alpha=self.a, beta=self.b), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)
        self.softplus = torch.nn.Softplus()

        print('parameters')
        print(dict(self.named_parameters()).keys())

        self.optimizer = torch.optim.Adam([self.Gu, self.Gi], lr=self.learning_rate)
        self.edges_optimizer = torch.optim.Adam([self.F], lr=self.edges_lr)

    def propagate_embeddings(self, evaluate=False):

        print(self.device)

        print(self.edge_features.device())
        print(self.F.get_device())
        self.F.to(self.device)
        self.edge_features.to(self.device)
        print(self.edge_features.device())
        print(self.F.get_device())

#        edge_embeddings_u_i = matmul(self.edge_features, self.F.to(self.device)) * (1 - self.b)
        edge_embeddings_u_i = matmul(self.edge_features.to(self.device), self.F)
        edge_embeddings_i_u = matmul(self.item_features.to(self.device), self.F)[self.items] * (1-self.a)

        ego_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        all_embeddings = [ego_embeddings]
        edge_embeddings = torch.cat([edge_embeddings_u_i, edge_embeddings_i_u], dim=0)

        for layer in range(0, self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children()
                    )[layer](all_embeddings[layer].to(self.device), self.edge_index.to(self.device),
                             edge_embeddings.to(self.device))]
            else:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer](all_embeddings[layer].to(self.device), self.edge_index.to(self.device),
                         edge_embeddings.to(self.device))]

        if evaluate:
            self.propagation_network.train()

        all_embeddings = sum([all_embeddings[k] * self.alpha[k] for k in range(len(all_embeddings))])
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        return gu, gi

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)
        xui = torch.sum(gamma_u * gamma_i, -1)
        return xui

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))  # + self.bi

    def train_step(self, batch):

        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos = self.forward(inputs=(gu[user[:, 0]], gi[pos[:, 0]]))
        xu_neg = self.forward(inputs=(gu[user[:, 0]], gi[neg[:, 0]]))
        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        bpr_loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.Gi, 2))
        features_reg_loss = self.l_w * torch.norm(self.F, 2)
        loss = bpr_loss + reg_loss

        # independence loss over the features within the same path
        if self.gamma > 0:
            n_edges = self.edge_features.size(0)
            n_selected_edges = int(n_edges * 0.01)
            selected_edges = random.sample(list(range(n_edges)), n_selected_edges)
            ind_loss = [torch.abs(torch.corrcoef(self.F[self.edge_features[e].storage._col])).sum() - len(
                self.edge_features[e].storage._col) for e in selected_edges]
            ind_loss = sum(ind_loss) / n_selected_edges
            ind_loss = ind_loss * self.gamma

        self.optimizer.zero_grad()
        self.edges_optimizer.zero_grad()

        loss.backward(retain_graph=True)
        if self.gamma > 0:
             ind_loss.backward(retain_graph=True)
        features_reg_loss.backward()

        self.optimizer.step()
        self.edges_optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
