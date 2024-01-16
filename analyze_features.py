import os
import pickle as pk
import numpy as np
import pandas as pd

dataset_name = 'yahoo_movies'


def feature_path(data_name, depth, npr):
    if depth == -1:
        depth = 'None'

    file_name = 'item_features' + str(npr) + '_entropy' + str(depth) + '.pk'
    return os.path.join('data', data_name, 'kgtore', file_name)


result = []
cols = ['npr', 'depth', 'avg_if', 'max_if', 'min_if']
for depth in [1, 2, 5, 10, 20, -1]:
    for npr in [1, 2, 5, 10, 20]:
        path = feature_path(data_name=dataset_name, depth=depth, npr=npr)
        if os.path.exists(path):
            with open(path, 'rb') as file:
                i_f = pk.load(file)
                n_features = i_f.size(dim=1)
                features_mean = i_f.nnz()/i_f.size(dim=0)
                prova = i_f.to_dense().numpy()
                s = np.sum(prova > 0, axis=1)
                data_result = [npr, depth, features_mean, s.max(), s.min()]
                result.append(data_result)

                # print(npr, depth, sum(s==0))
df = pd.DataFrame(result, columns=cols)
df.to_csv('item_features_analysis.tsv', sep='\t', index=False, decimal=',')
