from data_preprocessing.filters.knowledge import *
from data_preprocessing.filters.dataset import UserItemIterativeKCore


class KGToreFilter(FilterPipeline):

    def __init__(self, dataset, kg: pd.DataFrame, linking: pd.DataFrame, core, **kwargs):
        print('\n-- KGTORE --')
        filters = [RemoveNoisyTriples, FilterKG, KGDatasetAlignment, DatasetKGAlignment, UserItemIterativeKCore, MapKG]
        super(KGToreFilter, self).__init__(filters,
                                           dataset=dataset,
                                           kg=kg,
                                           linking=linking,
                                           core=core)

    @property
    def kg(self):
        return self._kwargs['kg']

    def filter_engine(self):
        n_ratings = len(self._kwargs['dataset'])
        super(KGToreFilter, self).filter_engine()
        new_n_ratings = len(self._kwargs['dataset'])
        self._flag = (n_ratings - new_n_ratings) == 0


class KaHFMFilter(FilterPipeline):

    def __init__(self, dataset, kg: pd.DataFrame, linking: pd.DataFrame, core, **kwargs):
        print('\n-- KAHFM --')
        filters = [KGFeaturesByFrequency, KGDatasetAlignment, DatasetKGAlignment, UserItemIterativeKCore, MapKG]
        super(KaHFMFilter, self).__init__(filters,
                                          dataset=dataset,
                                          kg=kg,
                                          linking=linking,
                                          core=core)

    def filter_engine(self):
        n_ratings = len(self._kwargs['dataset'])
        super(KaHFMFilter, self).filter_engine()
        new_n_ratings = len(self._kwargs['dataset'])
        self._flag = (n_ratings - new_n_ratings) == 0


class KGATFilter(FilterPipeline):

    def __init__(self, dataset, kg: pd.DataFrame, linking: pd.DataFrame, core, pred_kcore, **kwargs):
        print('\n-- KGAT --')
        filters = [PredKCore, KGDatasetAlignment, DatasetKGAlignment, MapKG]
        super(KGATFilter, self).__init__(filters,
                                         dataset=dataset,
                                         kg=kg,
                                         linking=linking,
                                         core=core,
                                         pred_kcore=pred_kcore)

    def filter_engine(self):
        n_ratings = len(self._kwargs['dataset'])
        super(KGATFilter, self).filter_engine()
        new_n_ratings = len(self._kwargs['dataset'])
        self._flag = (n_ratings - new_n_ratings) == 0


class KGINFilter(FilterPipeline):

    def __init__(self, dataset, kg: pd.DataFrame, linking: pd.DataFrame, core, **kwargs):
        print('\n-- KGIN --')
        filters = [KGIterativeKCore, KGDatasetAlignment, DatasetKGAlignment, MapKG]
        super(KGINFilter, self).__init__(filters,
                                         dataset=dataset,
                                         kg=kg,
                                         linking=linking,
                                         core=core)

    def filter_engine(self):
        n_ratings = len(self._kwargs['dataset'])
        super(KGINFilter, self).filter_engine()
        new_n_ratings = len(self._kwargs['dataset'])
        self._flag = (n_ratings - new_n_ratings) == 0


class KGFlexFilter(FilterPipeline):

    def __init__(self, dataset, kg: pd.DataFrame, linking: pd.DataFrame, core, **kwargs):
        print('\n-- KGFLEX --')
        filters = [FilterKG, KGDatasetAlignment, DatasetKGAlignment, UserItemIterativeKCore, MapKG, ItemFeatures]
        super(KGFlexFilter, self).__init__(filters,
                                           dataset=dataset,
                                           kg=kg,
                                           linking=linking,
                                           core=core)

    @property
    def kg(self):
        return self._kwargs['kg']

    def filter_engine(self):
        n_ratings = len(self._kwargs['dataset'])
        super(KGFlexFilter, self).filter_engine()
        new_n_ratings = len(self._kwargs['dataset'])
        self._flag = (n_ratings - new_n_ratings) == 0
