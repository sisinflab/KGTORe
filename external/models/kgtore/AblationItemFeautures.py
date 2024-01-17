import torch_sparse
import random
import pickle
import torch
import numpy as np

def create_random_item_features(n_items:int=10, min_features:int=1, max_features:int=10, seed=123):
    # random.sample(range(1, 100), 3) generate 3 samples without repetition
    np.random.seed(seed)
    random.seed(seed)
    i_f = {i: [value for value in random.sample(range(min_features - 1, max_features + 1), random.randint(min_features, max_features))] for i in range(n_items) }
    row_indices = [r for i in [[k] * len(v) for k, v in i_f.items()] for r in i]
    col_indices = [r for i in [v for _, v in i_f.items()] for r in i]
    item_features = torch_sparse.SparseTensor(row=torch.tensor(row_indices, dtype=torch.int64),
                                                   col=torch.tensor(col_indices, dtype=torch.int64),
                                                   value=torch.tensor([r for i in
                                                                       [torch.ones(len(v)) / len(v) for _, v in
                                                                        i_f.items()] for r in i],
                                                                      dtype=torch.float64))
    return item_features
    print()


create_item_features()