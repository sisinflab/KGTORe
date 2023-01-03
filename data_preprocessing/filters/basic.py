import pandas as pd
from data_preprocessing.filters import Filter


class KCore(Filter):
    def __init__(self, dataset: pd.DataFrame, kcore_column, core, **kwargs):
        super(KCore, self).__init__()
        self._dataset = dataset.copy()
        self._column = kcore_column
        self._core = core

    def filter_engine(self):
        print(f'{self.__class__.__name__}: {self._core}-core')
        print(f'{self.__class__.__name__}: filtering by column \'{self._column}\'')
        n_records = len(self._dataset)
        print(f'{self.__class__.__name__}: {n_records} transactions found')
        groups = self._dataset.groupby([self._column])
        self._dataset = groups.filter(lambda x: len(x) >= self._core)
        new_n_records = len(self._dataset)
        print(f'{self.__class__.__name__}: {n_records - new_n_records} transactions removed')
        print(f'{self.__class__.__name__}: {new_n_records} transactions retained')
        self._flag = (n_records - new_n_records) == 0

    def filter_output(self):
        self._output['dataset'] = self._dataset
        return self._output


class IterativeKCore(Filter):

    def __init__(self, dataset: pd.DataFrame, kcore_columns: list, core, **kwargs):
        super(IterativeKCore, self).__init__()
        self._dataset = dataset.copy()
        self._columns = kcore_columns
        self._core = core

    def filter_engine(self):

        print(f'{self.__class__.__name__}: iterative {self._core}-core')
        check = False
        while not check:
            checks = []
            for c in self._columns:
                f = KCore(dataset=self._dataset, kcore_column=c, core=self._core)
                out = f.filter()
                self._dataset = out['dataset']
                checks.append(f.flag)
            check = all(checks)
        self._flag = True

    def filter_output(self):
        self._output['dataset'] = self._dataset
        return self._output
