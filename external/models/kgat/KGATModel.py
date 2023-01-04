from abc import ABC
import torch
import numpy as np
import random
import dgl
from scipy import sparse as sp
from .Aggregator import Aggregator


class KGATModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_entities,
                 num_relations,
                 learning_rate,
                 embed_k,
                 kg_embed_k,
                 aggr,
                 l_w,
                 weight_size,
                 message_dropout,
                 kg_graph,
                 rows,
                 cols,
                 data,
                 random_seed,
                 name="KGAT",
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

        self.num_users = num_users
        self.embed_k = embed_k
        self.kg_embed_k = kg_embed_k
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.aggregator_type = aggr
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.layers = [self.embed_k] + list(weight_size)
        self.mess_dropout = message_dropout
        self.kg_graph = kg_graph

        self.all_hs = torch.LongTensor(rows).to(self.device)
        self.all_ts = torch.LongTensor(cols).to(self.device)
        self.all_rs = torch.LongTensor(data).to(self.device)
        self.matrix_size = torch.Size(
            [self.num_users + self.num_entities, self.num_users + self.num_entities]
        )

        self.A_in = (
            self.init_graph()
        )

        self.user_embedding = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        self.user_embedding.to(self.device)
        self.entity_embedding = torch.nn.Embedding(self.num_entities, self.embed_k)
        torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        self.entity_embedding.to(self.device)
        self.relation_embedding = torch.nn.Embedding(
            self.num_relations + 1, self.embed_k
        )
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)
        self.relation_embedding.to(self.device)
        self.trans_w = torch.nn.Embedding(
            self.num_relations + 1, self.embed_k * self.kg_embed_k
        )
        torch.nn.init.xavier_uniform_(self.trans_w.weight)
        self.trans_w.to(self.device)
        self.aggregator_layers = torch.nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(
                zip(self.layers[:-1], self.layers[1:])
        ):
            self.aggregator_layers.append(
                Aggregator(
                    input_dim, output_dim, self.mess_dropout, self.aggregator_type, self.device
                )
            )
        self.tanh = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def init_graph(self):
        adj_list = []
        for rel_type in range(1, self.num_relations, 1):
            edge_idxs = self.kg_graph.filter_edges(
                lambda edge: edge.data["relation_id"] == rel_type
            )
            sub_graph = (
                dgl.edge_subgraph(self.kg_graph, edge_idxs, preserve_nodes=True)
                    .adjacency_matrix(transpose=False, scipy_fmt="coo")
                    .astype("float")
            )
            rowsum = np.array(sub_graph.sum(1))
            with np.errstate(divide='ignore'):
                d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor(np.array([final_adj_matrix.row, final_adj_matrix.col]))
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings)
            norm_embeddings = torch.nn.functional.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.num_users, self.num_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h)
        pos_t_e = self.entity_embedding(pos_t)
        neg_t_e = self.entity_embedding(neg_t)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(
            r.size(0), self.embed_k, self.kg_embed_k
        )

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

    def generate_transE_score(self, hs, ts, r):
        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r]
        r_trans_w = self.trans_w.weight[r].view(
            self.embed_k, self.kg_embed_k
        )

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)

        kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)

        return kg_score

    def update_attentive_A(self):
        kg_score_list, row_list, col_list = [], [], []
        for rel_idx in range(1, self.num_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(
                self.all_hs[triple_index], self.all_ts[triple_index], rel_idx
            )
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        self.A_in = A_in

    def train_step(self, batch, batch_kg):

        # bprmf
        user, pos_item, neg_item = batch
        user_all_embeddings, entity_all_embeddings = self.forward()
        # u_embeddings = user_all_embeddings[user]
        # pos_embeddings = entity_all_embeddings[pos_item]
        # neg_embeddings = entity_all_embeddings[neg_item]
        u_embeddings = user_all_embeddings[np.squeeze(user)]
        pos_embeddings = entity_all_embeddings[np.squeeze(pos_item)]
        neg_embeddings = entity_all_embeddings[np.squeeze(neg_item)]
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        l2_loss = torch.norm(u_embeddings, 2) + torch.norm(pos_embeddings, 2) + torch.norm(neg_embeddings, 2)
        loss_bprmf = mf_loss + self.l_w * l2_loss

        # loss kg
        h, r, pos_t, neg_t = batch_kg
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(torch.tensor(h, dtype=torch.int64, device=self.device),
                                                            torch.tensor(r, dtype=torch.int64, device=self.device),
                                                            torch.tensor(pos_t, dtype=torch.int64, device=self.device),
                                                            torch.tensor(neg_t, dtype=torch.int64, device=self.device))
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = self.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = torch.norm(h_e, 2) + torch.norm(r_e, 2) + torch.norm(pos_t_e, 2) + torch.norm(neg_t_e, 2)
        loss_kg = kg_loss + self.l_w * kg_reg_loss

        loss = loss_bprmf + loss_kg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def predict(self, batch_user, items):
        user_all_embeddings, entity_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[batch_user]
        i_embeddings = entity_all_embeddings[items]
        # scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        scores = torch.matmul(u_embeddings, torch.transpose(i_embeddings, 0, 1))
        return scores

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
