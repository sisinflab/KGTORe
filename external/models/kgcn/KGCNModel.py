from abc import ABC
import torch
import numpy as np
import random


class KGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_entities,
                 num_relations,
                 learning_rate,
                 embed_k,
                 aggr,
                 l_w,
                 neighbor_sample_size,
                 kg_graph,
                 n_iter,
                 random_seed,
                 name="KGCN",
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

        self.embedding_size = embed_k
        self.aggregator_class = aggr
        self.reg_weight = l_w
        self.neighbor_sample_size = neighbor_sample_size
        self.num_users = num_users
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.batch_size = None
        self.n_iter = n_iter
        self.learning_rate = learning_rate

        self.user_embedding = torch.nn.Embedding(self.num_users, self.embedding_size)
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        self.user_embedding.to(self.device)
        self.entity_embedding = torch.nn.Embedding(self.num_entities, self.embedding_size)
        torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        self.entity_embedding.to(self.device)
        self.relation_embedding = torch.nn.Embedding(
            self.num_relations + 1, self.embedding_size
        )
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)
        self.relation_embedding.to(self.device)

        adj_entity, adj_relation = self.construct_adj(kg_graph)
        self.adj_entity, self.adj_relation = adj_entity.to(
            self.device
        ), adj_relation.to(self.device)

        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear_layers = torch.nn.ModuleList()
        for i in range(self.n_iter):
            self.linear_layers.append(
                torch.nn.Linear(
                    self.embedding_size
                    if not self.aggregator_class == "concat"
                    else self.embedding_size * 2,
                    self.embedding_size,
                )
            )
        self.linear_layers.to(self.device)
        self.ReLU = torch.nn.ReLU()
        self.Tanh = torch.nn.Tanh()

        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def construct_adj(self, kg_graph):
        kg_dict = dict()
        for triple in zip(kg_graph.row, kg_graph.data, kg_graph.col):
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            if head not in kg_dict:
                kg_dict[head] = []
            kg_dict[head].append((tail, relation))
            if tail not in kg_dict:
                kg_dict[tail] = []
            kg_dict[tail].append((head, relation))

        entity_num = kg_graph.shape[0]
        adj_entity = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        adj_relation = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        for entity in range(entity_num):
            if entity not in kg_dict.keys():
                adj_entity[entity] = np.array([entity] * self.neighbor_sample_size)
                adj_relation[entity] = np.array([0] * self.neighbor_sample_size)
                continue

            neighbors = kg_dict[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= self.neighbor_sample_size:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=False,
                )
            else:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=True,
                )
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

        return torch.from_numpy(adj_entity), torch.from_numpy(adj_relation)

    def get_neighbors(self, items):
        items = torch.unsqueeze(items, dim=1)
        entities = [items]
        relations = []
        for i in range(self.n_iter):
            index = torch.flatten(entities[i])
            neighbor_entities = torch.index_select(self.adj_entity, 0, index).reshape(
                self.batch_size, -1
            )
            neighbor_relations = torch.index_select(
                self.adj_relation, 0, index
            ).reshape(self.batch_size, -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            user_embeddings = user_embeddings.reshape(
                self.batch_size, 1, 1, self.embedding_size
            )  # [batch_size, 1, 1, dim]
            user_relation_scores = torch.mean(
                user_embeddings * neighbor_relations, dim=-1
            )  # [batch_size, -1, n_neighbor]
            user_relation_scores_normalized = self.softmax(
                user_relation_scores
            )  # [batch_size, -1, n_neighbor]

            user_relation_scores_normalized = torch.unsqueeze(
                user_relation_scores_normalized, dim=-1
            )  # [batch_size, -1, n_neighbor, 1]
            neighbors_aggregated = torch.mean(
                user_relation_scores_normalized * neighbor_vectors, dim=2
            )  # [batch_size, -1, dim]
        else:
            neighbors_aggregated = torch.mean(
                neighbor_vectors, dim=2
            )  # [batch_size, -1, dim]
        return neighbors_aggregated

    def aggregate(self, user_embeddings, entities, relations):
        entity_vectors = [self.entity_embedding(i) for i in entities]
        relation_vectors = [self.relation_embedding(i) for i in relations]

        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = (
                    self.batch_size,
                    -1,
                    self.neighbor_sample_size,
                    self.embedding_size,
                )
                self_vectors = entity_vectors[hop]
                neighbor_vectors = entity_vectors[hop + 1].reshape(shape)
                neighbor_relations = relation_vectors[hop].reshape(shape)

                neighbors_agg = self.mix_neighbor_vectors(
                    neighbor_vectors, neighbor_relations, user_embeddings
                )  # [batch_size, -1, dim]

                if self.aggregator_class == "sum":
                    output = (self_vectors + neighbors_agg).reshape(
                        -1, self.embedding_size
                    )  # [-1, dim]
                elif self.aggregator_class == "neighbor":
                    output = neighbors_agg.reshape(-1, self.embedding_size)  # [-1, dim]
                elif self.aggregator_class == "concat":
                    # [batch_size, -1, dim * 2]
                    output = torch.cat([self_vectors, neighbors_agg], dim=-1)
                    output = output.reshape(
                        -1, self.embedding_size * 2
                    )  # [-1, dim * 2]
                else:
                    raise Exception("Unknown aggregator: " + self.aggregator_class)

                output = self.linear_layers[i](output)
                # [batch_size, -1, dim]
                output = output.reshape(self.batch_size, -1, self.embedding_size)

                if i == self.n_iter - 1:
                    vector = self.Tanh(output)
                else:
                    vector = self.ReLU(output)

                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        item_embeddings = entity_vectors[0].reshape(
            self.batch_size, self.embedding_size
        )

        return item_embeddings

    def forward(self, user, item):
        self.batch_size = item.shape[0]
        user_e = self.user_embedding(user.to(self.device)).to(self.device)
        entities, relations = self.get_neighbors(item.to(self.device))
        item_e = self.aggregate(user_e, entities, relations)
        return user_e, item_e

    def train_step(self, batch):
        user, pos_item, neg_item = batch

        user_e, pos_item_e = self.forward(torch.tensor(user[:, 0], dtype=torch.int64),
                                          torch.tensor(pos_item[:, 0], dtype=torch.int64))
        user_e, neg_item_e = self.forward(torch.tensor(user[:, 0], dtype=torch.int64),
                                          torch.tensor(neg_item[:, 0], dtype=torch.int64))

        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1)

        predict = torch.cat((pos_item_score, neg_item_score))
        target = torch.zeros(len(user) * 2, dtype=torch.float32).to(self.device)
        target[: len(user)] = 1
        rec_loss = self.bce_loss(predict, target)

        l2_loss = torch.norm(user_e, 2) + torch.norm(pos_item_e, 2) + torch.norm(neg_item_e, 2)
        loss = rec_loss + self.reg_weight * l2_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def predict(self, user, item, batch_user, batch_item, **kwargs):
        self.batch_size = item.shape[0]
        user_e = self.user_embedding(user.to(self.device)).to(self.device)
        entities, relations = self.get_neighbors(item.to(self.device))
        item_e = self.aggregate(user_e, entities, relations)
        return torch.mul(user_e, item_e.to(self.device)).sum(dim=1).view(batch_user, batch_item)

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), torch.tensor(preds).to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
