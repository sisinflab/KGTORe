import pandas as pd
import ast
import os
import numpy as np

filename_list = ['decision_path_entropy_1.tsv', 'decision_path_entropy_5.tsv', 'decision_path_entropy_10.tsv',
                 'decision_path_gini_1.tsv', 'decision_path_gini_5.tsv', 'decision_path_gini_10.tsv']

filename_list2 = ['./external/models/egcfv2_Salv/data/decision_path_entropy_1.tsv',
                  './external/models/egcfv2_Salv/data/decision_path_entropy_5.tsv',
                  './external/models/egcfv2_Salv/data/decision_path_entropy_10.tsv',
                  './external/models/egcfv2_Salv/data/decision_path_gini_1.tsv',
                  './external/models/egcfv2_Salv/data/decision_path_gini_5.tsv',
                  './external/models/egcfv2_Salv/data/decision_path_gini_10.tsv']


def load_dataset(filename=filename_list[0], default_path=True):
    """
    Load the tsv dataset - [ user | feature_path | item ]
    :param filename: name of the dataset
    :return: pandas dataset
    """
    if default_path:
        path = './external/models/egcfv2/data/' + filename
    else:
        path = filename
    dp = pd.read_csv(path, sep='\t')
    dp = dp.astype({"user": "int", "item": "int"})
    dp['feature_path'] = dp['feature_path'].apply(lambda x: ast.literal_eval(x))
    return dp


def load_dataset_all(filename_list=filename_list, default_path=True):
    """
     :param filename: list of filenames paths es: ['file_name_1', 'file_file_name2']
    :return: dictionary of dataframes key=file_name, value = file_dataframe es: ['file_name_1': dg_1, 'file_name_2': df2]
    """
    decision_path_dict = dict()
    if default_path:
        for i in range(len(filename_list)):
            decision_path_dict[filename_list[i]] = load_dataset(filename_list[i])
    else:
        for i in range(len(filename_list)):
            name = filename_list[i].split('/')[-1]
            decision_path_dict[name] = load_dataset(filename_list[i], default_path=False)
    return decision_path_dict


def load_feature_map(filename='./external/models/egcfv2_Salv/data/new_feature_map.csv'):
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df['feature'] = df['feature'].apply(lambda x: ast.literal_eval(x))  # [feature, val_numerico_mappato]
        new_feature_map = dict(
            zip(df.feature, df.mapped_value))  # { ('23322','43455') : 0, ('3434232',..):1 ..}
        feature_set = set()
        feature_set.update(df.feature)
    else:
        new_feature_map = dict()
        feature_set = set()
    return new_feature_map, feature_set


def prova(dataset):
    abs_feature_path = list()
    for i in range(len(dataset)):
        abs_feature_path.append(list(map(abs, dataset.iloc[i, 2])))
    dataset['abs_feature_path'] = abs_feature_path
    # dataset = [user item feature_path abs_feature_path]
    # es [1  10432  [-1, 2, 5, -7, 99] ] --> [1  10432  [-1, 2, 5, -7, 99] [1, 2, 5, 7, 99] ]
    '''dataset = dataset.join(pd.DataFrame(mlb.fit_transform(dataset.pop('feature_path')),
                          columns=mlb.classes_,
                          index=dataset.index))'''
    return dataset

def one_hot_dataset(dataset):
    df = dataset.explode('feature_path')
    df['f1'] = abs(df.feature_path)
    df['f2'] = np.sign(df.feature_path)
    df = df.pivot(['user', 'item'], 'f1', 'f2').fillna(0).reset_index()
    '''
        from this:
                user       item     features
    0     a            1        [2]
    1     a            2        [-2, -1]
    2     b            1        [-137, -1, 2]
    3     b            3        [-137, 2, 1]
        to this:
    f2 user  item  1  2  137
    0     a     1  0  1    0
    1     a     2 -1 -1    0
    2     b     1 -1  1   -1
    3     b     3  1  1   -1
    '''
    return df


def add_missing_values(dataframe, fmap):
    for i in range(1, len(fmap) + 1):  # features numerate a partire da 1
        if i not in dataframe.columns:
            dataframe[i] = 0
    return dataframe


''' final dataset:
df = load_dataset(filename_list[0])
fmap, _ = load_feature_map()
df = one_hot_dataset(df)
df = add_missing_values(df, fmap)
3- '''