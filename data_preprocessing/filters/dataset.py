import numpy as np
import pandas as pd

from data_preprocessing.filters.basic import IterativeKCore
from data_preprocessing.filters import Filter
from sklearn.model_selection import train_test_split as split


class UserItemIterativeKCore(IterativeKCore):
    def __init__(self, dataset, core, **kwargs):
        super(UserItemIterativeKCore, self).__init__(dataset=dataset, core=core, kcore_columns=['u', 'i'], **kwargs)


class Binarize(Filter):
    def __init__(self, dataset, threshold):
        super(Binarize, self).__init__()
        self._dataset = dataset.copy()
        self._binary_threshold = threshold

    def filter_engine(self):
        n_ratings = len(self._dataset)
        print(f'{self.__class__.__name__}: {n_ratings} transactions found')
        retained = self._dataset.r >= self._binary_threshold
        self._dataset = self._dataset[retained]
        self._dataset = self._dataset[['u', 'i']]
        new_n_ratings = len(self._dataset)
        self._dataset['r'] = [1] * new_n_ratings
        print(f'{self.__class__.__name__}: {n_ratings - new_n_ratings} transactions removed')
        print(f'{self.__class__.__name__}: {new_n_ratings} transactions retained')
        self._flag = (n_ratings - new_n_ratings) == 0

    def filter_output(self):
        return {'dataset': self._dataset}


class Splitter(Filter):
    def __init__(self, data, test_ratio=0, val_ratio=0):
        super(Splitter, self).__init__()
        self._dataset = data.copy()
        self._test_ratio = test_ratio
        self._val_ratio = val_ratio

        self._train = pd.DataFrame()
        self._test = pd.DataFrame()
        self._val = pd.DataFrame()

    def filter_engine(self):

        for u in self._dataset.iloc[:, 0].unique():
            u_df = self._dataset[self._dataset.iloc[:, 0] == u]
            u_train, u_test = split(u_df, test_size=self._test_ratio, random_state=42)
            u_train, u_val = split(u_train, test_size=self._val_ratio, random_state=42)
            self._train = pd.concat([self._train, u_train], axis=0, ignore_index=True)
            self._test = pd.concat([self._test, u_test], axis=0, ignore_index=True)
            self._val = pd.concat([self._val, u_val], axis=0, ignore_index=True)
        self._flag = True

    def filter_output(self):
        return {'train': self._train,
                'test': self._test,
                'val': self._val}
