import pickle

import pandas as pd
import torch
from torch_sparse import SparseTensor
import numpy as np
from collections import Counter

def LoadEdgeFeatures(path, item_features_path, transactions):
    edge_features = pd.read_csv(path, sep='\t', header=None)
    edge_features.columns = ['user', 'item', 'feature', 'val', 'interaction']
    print('loading item features')
    with open(item_features_path, 'rb') as file:
        item_features = pickle.load(file)
    print(f'item features loaded from \'{item_features_path}\'')

    # weighted tree order
    #ss = edge_features.groupby(['user', 'item']).transform(
    #     lambda x: (np.array(range(len(x), 0, -1)) / sum(np.array(range(len(x), 0, -1)))))
    #edge_features['val'] = abs(edge_features['val'])
    #edge_features['val'] = ss['val'] * edge_features['val']

    # edge_features['val'] = abs(edge_features['val'])
    print()
    return SparseTensor(row=torch.tensor(edge_features['interaction'], dtype=torch.int64),
                                      col=torch.tensor(edge_features['feature'].astype(int).to_numpy(),
                                                       dtype=torch.int64),
                                      value=torch.tensor(edge_features['val'].astype(float).to_numpy(),
                                                         dtype=torch.float32),
                                      sparse_sizes=(transactions, edge_features['feature'].nunique()),), item_features
