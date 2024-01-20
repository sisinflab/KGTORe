import torch_sparse
import random
import pickle
import torch
import numpy as np
import pandas as pd
import itertools


def build_if(knowledge_graph:pd.DataFrame, public_items:dict) -> dict:
    i_f = knowledge_graph.copy()
    i_f['subject'] = i_f['subject'].map(public_items)
    feature_column = i_f['predicate'].astype(str) + '-' + i_f['object'].astype(str)
    i_f = pd.concat([i_f['subject'], feature_column], axis=1)
    i_f.columns = ['subject', 'feature']
    feature_to_private = {f: i for i, f in enumerate(i_f['feature'].unique(), 1)}
    i_f['feature'] = i_f['feature'].map(feature_to_private)  # da df a mapping interno
    i_f_dict = i_f.groupby('subject')['feature'].apply(set).to_dict()
    return i_f_dict

def create_random_item_features(n_items: int = 10, min_features: int = 1, max_features: int = 10,
                                seed=123) -> torch_sparse.SparseTensor:
    # random.sample(range(1, 100), 3) generate 3 samples without repetition
    np.random.seed(seed)
    random.seed(seed)
    i_f = {i: [value for value in
               random.sample(range(min_features - 1, max_features), random.randint(min_features, max_features))] for
           i in range(n_items)}
    row_indices = [r for i in [[k] * len(v) for k, v in i_f.items()] for r in i]
    col_indices = [r for i in [v for _, v in i_f.items()] for r in i]
    item_features = torch_sparse.SparseTensor(row=torch.tensor(row_indices, dtype=torch.int64),
                                              col=torch.tensor(col_indices, dtype=torch.int64),
                                              value=torch.tensor([r for i in
                                                                  [torch.ones(len(v)) / len(v) for _, v in
                                                                   i_f.items()] for r in i],
                                                                 dtype=torch.float64))
    return item_features


# def create_shuffled_item_features(old_item_features: torch_sparse.SparseTensor, n_items: int = 10,
#                                   min_features: int = 1, max_features: int = 10, seed=123):
#     shuddled_if = old_item_features.to_dense().clone()
#     shuffled_if = shuddled_if[torch.randperm(shuddled_if.size()[0])]
#
#     return None

def create_shuffled_item_features(old_item_features: torch_sparse.SparseTensor,
                                  seed: int = 123) -> torch_sparse.SparseTensor:
    np.random.seed(seed)
    random.seed(seed)
    rows = old_item_features.storage.row()
    items = old_item_features.storage.row().unique()
    min_features = 0 # items.min()
    max_features = items.max()
    sub = random.sample(range(min_features, max_features + 1), 10)
    sub_dict = {i: sub[i] for i in range(len(items))}
    replacer = sub_dict.get
    new_rows = torch.tensor([replacer(int(n), int(n)) for n in rows], dtype=torch.int64)

    # new_rows = torch.tensor([int(i) for i in new_rows], dtype=torch.int64)
    cols = old_item_features.storage.col()
    values = old_item_features.storage.value()
    item_features = torch_sparse.SparseTensor(row=new_rows,
                                              col=cols,
                                              value=values)
    return item_features


def create_random_real_item_features(knowledge_graph:pd.DataFrame, public_items:dict, seed:int=123) -> torch_sparse.SparseTensor:
    np.random.seed(seed)
    random.seed(seed)
    i_f = build_if(knowledge_graph=knowledge_graph, public_items=public_items)
    i_f = {i: [value for value in
               random.sample(i_f[i], random.randint(1, len(i_f[i])))] for
           i in i_f.keys()}
    all_features = set(itertools.chain(*i_f.values()))
    feature_to_private = {f: private for private, f in enumerate(all_features, 0)}
    i_f = {i: set(list(map(feature_to_private.get, i_f[i]))) for i in i_f.keys()}
    row_indices = [r for i in [[k] * len(v) for k, v in i_f.items()] for r in i]
    col_indices = [r for i in [v for _, v in i_f.items()] for r in i]
    item_features = torch_sparse.SparseTensor(row=torch.tensor(row_indices, dtype=torch.int64),
                                              col=torch.tensor(col_indices, dtype=torch.int64),
                                              value=torch.tensor([r for i in
                                                                  [torch.ones(len(v)) / len(v) for _, v in
                                                                   i_f.items()] for r in i],
                                                                 dtype=torch.float64))
    return item_features







