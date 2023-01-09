def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
import scipy
import os.path
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import numpy as np
from torch_sparse import SparseTensor

seed = 0


class DecisionPaths():
    def __init__(self, interactions, u_i_dict, kg, public_items, public_users, transaction, device, df_name, npr=10, criterion='entropy'):
        self.interactions = interactions
        self.public_items = public_items
        self.public_users = public_users
        self.transaction = transaction
        self.u_i_dict = u_i_dict
        self.device = device
        self.npr = npr
        self.criterion = criterion
        self.dataset_name = df_name
        self._feature_to_private = None
        self.i_f = None
        self.train_dict = None
        self.edge_features = list()
        self.build_if(kg)
        self.build_decision_paths()  # for feature dev.

    def save_edge_features_df(self, edge_feature_df):
        name = 'decision_path' + str(self.npr) + "_" + str(self.criterion) + ".tsv"
        dataset_path = os.path.abspath(os.path.join('./data', self.dataset_name, 'kgtore', name))
        edge_feature_df.to_csv(dataset_path, sep='\t', header=False, index=False)

    def build_if(self, kg):
        i_f = kg
        i_f['subject'] = i_f['subject'].map(self.public_items)
        feature_column = i_f['predicate'].astype(str) + '-' + i_f['object'].astype(str)
        i_f = pd.concat([i_f['subject'], feature_column], axis=1)
        i_f.columns = ['subject', 'feature']
        self._feature_to_private = {f: i for i, f in enumerate(i_f['feature'].unique(), 1)}
        i_f['feature'] = i_f['feature'].map(self._feature_to_private)  # da df a mapping interno
        self.i_f = i_f.groupby('subject')['feature'].apply(set).to_dict()
        # self._data.i_train_dict = {u1: {i1: rating, i2:rating, ..}, u2: {ix:rating, ..}}

    def create_edge_features_matrix(self):
        edge_features = pd.DataFrame(self.edge_features)
        self.save_edge_features_df(edge_features)
        edge_features.columns = ['user', 'item', 'feature']
        edge_features['val'] = np.sign(edge_features['feature'])
        edge_features['feature'] = np.abs(edge_features['feature'])
        new_mapping = {p: pnew for pnew, p in enumerate(edge_features['feature'].unique())}
        edge_features['feature'] = edge_features['feature'].map(new_mapping)
        private_to_feature = {p: f for f, p in self._feature_to_private.items()}
        feature_to_private = {private_to_feature[p]: pnew for p, pnew in new_mapping.items()}
        # feature_to_private = {f: new_mapping[p] for f, p in self._feature_to_private.items()}
        self._feature_to_private = feature_to_private
        # reindex by interaction
        indices = edge_features.groupby(['user', 'item']).size().reset_index(name='Freq')
        index_list = [i for i in indices.index for z in range(indices.iloc[i, -1])]
        edge_features.index = index_list
        self.edge_features = SparseTensor(row=torch.tensor(edge_features.index, dtype=torch.int64),
                                          col=torch.tensor(edge_features['feature'].astype(int).to_numpy(),dtype=torch.int64),
                                          value=torch.tensor(edge_features['val'].astype(int).to_numpy(),dtype=torch.int64),
                                          sparse_sizes=(self.transaction, edge_features['feature'].nunique())).to(self.device)

    def build_decision_paths(self):
        criterion = self.criterion
        users = set(self.interactions.keys())
        items = set(self.i_f.keys())
        npr = self.npr

        def create_user_df(positive_items, negative_items, i_f, npr):
            negatives_len = npr * len(positive_items)  # n° item negativi che vogliamo considerare
            if len(positive_items) * npr <= len(negative_items):
                neg_items = random.sample(list(negative_items), k=negatives_len)
            else:
                ratio = len(negative_items) // len(positive_items)
                neg_items = random.sample(list(negative_items), k=ratio * len(positive_items)) if ratio > 0 else list(negative_items)
                neg_items.extend(random.choices(list(negative_items), k=negatives_len - len(neg_items)))

            all_items = list()
            all_items.extend(list(positive_items))
            all_items.extend(list(neg_items))
            mlb = MultiLabelBinarizer()
            d = {k: i_f[k] for k in all_items}
            df = pd.DataFrame(mlb.fit_transform(d.values()), columns=mlb.classes_)
            df['item_id'] = d.keys()
            df['positive'] = df['item_id'].isin(positive_items).astype(int)
            return df

        def create_user_tree(df, criterion):
            clf = DecisionTreeClassifier(criterion=criterion, class_weight={1: npr, 0: 1}, random_state=seed)
            X = scipy.sparse.csr_matrix(df.iloc[:, :-2].values)
            y = df.iloc[:, -1].values
            clf.fit(X, y)
            return clf

        def retrieve_decision_paths(df, clf, u, u_i_dict):
            #full_positive_df = df[df["positive"] == 1]
            full_positive_df = df.iloc[pd.Index(df['item_id']).get_indexer(u_i_dict[u])]
            csr = scipy.sparse.csr_matrix(full_positive_df.iloc[:, :-2].values)
            decision_path = clf.decision_path(csr)
            # decision_path = clf.decision_path(full_positive_df.iloc[:, :-2])
            # decision_path_dict = dict() # v1 old
            u_dp = list()
            for i in range(0, full_positive_df.shape[0]):
                sample_no = i  # riga_sample iesimo
                dp_i = decision_path.indices[decision_path.indptr[sample_no]: decision_path.indptr[sample_no + 1]]
                a = clf.tree_.feature[dp_i][clf.tree_.feature[dp_i] > 0]
                feature_is_present = full_positive_df.iloc[sample_no, a]
                feature_is_present = feature_is_present.replace(0, -1)
                final_dp_feature = list(feature_is_present.index.astype(int) * feature_is_present)
                # decision_path_dict[full_positive_df.iloc[sample_no, -2]] = final_dp_feature v1 old
                u_dp.extend([[u, full_positive_df.iloc[sample_no, -2], j] for j in final_dp_feature])  # u_dp = [ [user, itemid, 1stf], [user, itemid, 2nfeat], .., [user2, itemn, f1], ..]
            return u_dp

        print("Building decision trees")
        for u in tqdm(self.u_i_dict.keys()):
        # for u in [1, 2, 3]:
            df = create_user_df(set(self.interactions[u].keys()),
                                set.difference(items, set(self.interactions[u].keys())),
                                self.i_f,
                                npr)
            clf = create_user_tree(df, criterion)
            u_dp = retrieve_decision_paths(df, clf, u, self.u_i_dict)
            self.edge_features.extend(u_dp)
            # u_dp_dict = retrieve_decision_paths(df, clf) # v1 old
            # self.edge_features.extend([[u, item, u_dp_dict[item]] for item in u_dp_dict.keys()]) # v1 old
        self.create_edge_features_matrix()




