import pandas as pd
import torch
from torch_sparse import SparseTensor
import numpy as np
from collections import Counter

def LoadEdgeFeatures(path, transactions):
    edge_features = pd.read_csv(path, sep='\t', header=None)
    edge_features.columns = ['user', 'item', 'feature']
    edge_features['val'] = np.sign(edge_features['feature'])
    edge_features['feature'] = np.abs(edge_features['feature'])
    new_mapping = {p: pnew for pnew, p in enumerate(edge_features['feature'].unique())}
    edge_features['feature'] = edge_features['feature'].map(new_mapping)
    # reindex by interaction
    groups = edge_features.groupby(['user', 'item'])
    # edge_features['val'] = groups['val'].apply(lambda x: x/len(x))
    indices = groups.size().reset_index(name='Freq')
    index_list = [i for i in indices.index for z in range(indices.iloc[i, -1])]
    edge_features.index = index_list
    counted = Counter(index_list)
    val2 = [v for i, v in counted.items() for _ in range(v)]
    edge_features['val'] = edge_features['val'] / val2

    # weighted tree order
    #ss = edge_features.groupby(['user', 'item']).transform(
    #     lambda x: (np.array(range(len(x), 0, -1)) / sum(np.array(range(len(x), 0, -1)))))
    #edge_features['val'] = abs(edge_features['val'])
    #edge_features['val'] = ss['val'] * edge_features['val']

    # edge_features['val'] = abs(edge_features['val'])

    return SparseTensor(row=torch.tensor(edge_features.index, dtype=torch.int64),
                                      col=torch.tensor(edge_features['feature'].astype(int).to_numpy(),
                                                       dtype=torch.int64),
                                      value=torch.tensor(edge_features['val'].astype(float).to_numpy(),
                                                         dtype=torch.float32),
                                      sparse_sizes=(transactions, edge_features['feature'].nunique()),)
