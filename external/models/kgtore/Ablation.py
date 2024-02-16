import torch
import torch_sparse
import numpy as np
import random

def get_statistic(item_features: torch_sparse.SparseTensor) -> dict:
    result = dict()
    items, occurence = item_features.storage.row().unique(return_counts=True)  # occurrence == how many feature each item has
    items_no_f = dict(zip(items.tolist(), occurence.tolist()))  # key = itemsId, value=n of features
    # features = item_features.storage.col().unique()
    # min_f = min(item_features.storage.col())
    max_f = max(item_features.storage.col())  # feautures = from 0 to max_f
    result['items'] = items_no_f
    result['max_f'] = max_f
    return result




def build_random_item_features(item_features: torch_sparse.SparseTensor, seed:int = 123) -> torch_sparse.SparseTensor:
    np.random.seed(seed)
    random.seed(seed)
    stat = get_statistic(item_features)
    i_f = {i: [value for value in
               random.sample(range(0, stat['max_f'] + 1), no_f)] for
           i, no_f in stat['items'].items()}
    row_indices = [r for i in [[k] * len(v) for k, v in i_f.items()] for r in i]
    col_indices = [r for i in [v for _, v in i_f.items()] for r in i]
    item_features = torch_sparse.SparseTensor(row=torch.tensor(row_indices, dtype=torch.int64),
                                              col=torch.tensor(col_indices, dtype=torch.int64),
                                              value=torch.tensor([r for i in
                                                                  [torch.ones(len(v)) / len(v) for _, v in
                                                                   i_f.items()] for r in i],
                                                                 dtype=torch.float64))
    return item_features