import pandas as pd
import torch
from torch_sparse import SparseTensor
import numpy as np

def LoadEdgeFeatures(path, transactions):
    edge_features = pd.read_csv(path, sep='\t', header=None)
    edge_features.columns = ['user', 'item', 'feature']
    edge_features['val'] = np.sign(edge_features['feature'])
    edge_features['feature'] = np.abs(edge_features['feature'])
    new_mapping = {p: pnew for pnew, p in enumerate(edge_features['feature'].unique())}
    edge_features['feature'] = edge_features['feature'].map(new_mapping)
    # reindex by interaction
    indices = edge_features.groupby(['user', 'item']).size().reset_index(name='Freq')
    index_list = [i for i in indices.index for z in range(indices.iloc[i, -1])]
    edge_features.index = index_list
    return SparseTensor(row=torch.tensor(edge_features.index, dtype=torch.int64),
                                      col=torch.tensor(edge_features['feature'].astype(int).to_numpy(),
                                                       dtype=torch.int64),
                                      value=torch.tensor(edge_features['val'].astype(int).to_numpy(),
                                                         dtype=torch.int64),
                                      sparse_sizes=(transactions, edge_features['feature'].nunique()))
