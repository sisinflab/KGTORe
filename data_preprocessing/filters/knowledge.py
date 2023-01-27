from data_preprocessing.filters import *
from data_preprocessing.filters.basic import IterativeKCore, KCore
from collections import Counter
import pandas as pd
import numpy as np


class FilterKG(Filter):
    def __init__(self, kg: pd.DataFrame, rare_obj_threshold=3, min_obj_threshold=10, **kwargs):
        super(FilterKG, self).__init__()
        self._kg = kg.copy()
        self._obj_threshold = rare_obj_threshold
        self._min_obj_threshold = min_obj_threshold

    def filter_engine(self):
        triples = len(self._kg)
        obj_counter = Counter(self._kg.o)
        filter_rare_objs = [o for o, c in obj_counter.items() if c < self._obj_threshold]
        filter_common_objs = [o for o, c in obj_counter.items() if c > self._min_obj_threshold]

        preds_one_app = set(self._kg[self._kg.o.isin(filter_rare_objs)].p.unique())
        preds_more_one_app = set(self._kg[self._kg.o.isin(filter_common_objs)].p.unique())

        to_remove_preds = set.difference(preds_one_app, preds_more_one_app)
        self._kg = self._kg[~self._kg.p.isin(to_remove_preds)]
        new_triples = len(self._kg)
        print(f'{self.__class__.__name__}: {triples - new_triples} triples removed')
        self._flag = triples - new_triples == 0

    def filter_output(self):
        return {'kg': self._kg}


class KGFeaturesByFrequency(Filter):
    def __init__(self, kg, threshold=0.93, **kwargs):
        super(KGFeaturesByFrequency, self).__init__()
        self._kg = kg
        self._threshold = threshold

    def filter_engine(self):
        n_triples = len(self._kg)
        n_subjects = self._kg.s.nunique()
        self._kg = self._kg.groupby(['p', 'o']).filter(lambda x: (1 - len(x) / n_subjects) <= self._threshold)
        n_new_triples = len(self._kg)
        removed_triples = n_triples - n_new_triples
        print(f'{self.__class__.__name__}: {removed_triples} triples removed from the knowledge graph')
        self._flag = removed_triples == 0

    def filter_output(self):
        return {'kg': self._kg}


class MapKG(Filter):

    def __init__(self, dataset, kg, linking, **kwargs):
        super(MapKG, self).__init__()
        self._dataset = dataset.copy()
        self._kg = kg.copy()
        self._map = linking.copy()

    def filter_engine(self):
        # entities = set.union(set(self._kg.s), set(self._kg.o))
        entities = np.unique(np.concatenate([self._kg.s, self._kg.o]))

        predicates = self._kg.p.unique()

        predicates_mapping = dict(zip(predicates, range(len(predicates))))
        uri_item_map = dict(zip(self._map.e, self._map.i))

        # item-entities with the same id of the items
        next_id = max(uri_item_map.values())
        for ent in entities:
            if ent not in uri_item_map:
                next_id += 1
                uri_item_map[ent] = next_id

        self._kg.s = self._kg.s.map(uri_item_map)
        self._kg.o = self._kg.o.map(uri_item_map)
        self._kg.p = self._kg.p.map(predicates_mapping)

        entities_mapping = pd.DataFrame(uri_item_map.items(), columns=['uri', 'entity_id'])
        predicates_mapping = pd.DataFrame(predicates_mapping.items(), columns=['uri', 'predicate_id'])
        mapping = pd.DataFrame(zip(self._map.i, self._map.i), columns=['item', 'entity'])

        self._output = {
            'kg': self._kg,
            'entities_mapping': entities_mapping,
            'predicates_mapping': predicates_mapping,
            'linking': mapping
        }
        self._flag = True


class KGDatasetAlignment(Filter):

    def __init__(self, dataset: pd.DataFrame, kg: pd.DataFrame, linking: pd.DataFrame, **kwargs):
        super(KGDatasetAlignment, self).__init__()
        # loading
        self._dataset = dataset.copy()
        self._kg = kg.copy()
        self._map = linking.copy()

    def filter_engine(self):
        items = set(self._dataset.i.unique())
        n_items = len(items)
        uri_item_map = dict(zip(self._map.e, self._map.i))

        kg_items = set(map(uri_item_map.get, self._kg.s.unique()))
        if None in kg_items:
            kg_items.remove(None)

        common_items = set.intersection(items, kg_items)

        self._dataset = self._dataset[self._dataset.i.isin(common_items)]
        new_n_items = self._dataset.i.nunique()
        print(f'{self.__class__.__name__}: {n_items - new_n_items} items removed from the dataset')
        self._flag = n_items - new_n_items == 0

    def filter_output(self):
        return {'dataset': self._dataset}


class DatasetKGAlignment(Filter):

    def __init__(self, dataset: pd.DataFrame, kg: pd.DataFrame, linking: pd.DataFrame, **kwargs):
        super(DatasetKGAlignment, self).__init__()
        # loading
        self._dataset = dataset.copy()
        self._kg = kg.copy()
        self._map = linking.copy()

    def filter_engine(self):
        entities = set(self._kg.s.unique())
        n_entities = len(entities)
        item_uri_map = dict(zip(self._map.i, self._map.e))

        dataset_entities = set(map(item_uri_map.get, self._dataset.i.unique()))
        if None in dataset_entities:
            dataset_entities.remove(None)

        common_entities = set.intersection(entities, dataset_entities)

        self._kg = self._kg[self._kg.s.isin(common_entities)]

        new_n_entities = self._kg.s.nunique()
        print(f'{self.__class__.__name__}: {n_entities - new_n_entities} entities removed from the knowledge graph')
        self._flag = n_entities - new_n_entities == 0

    def filter_output(self):
        return {'kg': self._kg}


class KGIterativeKCore(IterativeKCore):
    def __init__(self, kg: pd.DataFrame, core: int, **kwargs):
        super(KGIterativeKCore, self).__init__(dataset=kg, core=core, kcore_columns=['s', 'o'])

    def filter_output(self):
        return {'kg': self._dataset}


class PredKCore(KCore):
    def __init__(self, kg: pd.DataFrame, pred_kcore, **kwargs):
        super(PredKCore, self).__init__(dataset=kg,
                                        kcore_column='p',
                                        core=pred_kcore)

    def filter_output(self):
        return {'kg': self._dataset}


class ItemFeatures(Filter):
    def __init__(self, dataset, kg, linking, **kwargs):
        self._dataset = dataset.copy()
        self._kg = kg.copy()
        self._map = linking.copy()
        super(ItemFeatures, self).__init__()

    def filter_engine(self):
        item_features = pd.DataFrame(zip(self._kg.s, zip(self._kg.p, self._kg.o)), columns=['i', 'f'])
        item_entity_map = dict(zip(self._map.entity, self._map.item))
        # map entities with items
        item_features.i = item_features.i.map(item_entity_map)
        self._output['item_features'] = item_features
        self._flag = True


class LinkingCleaning(Filter):
    def __init__(self, linking, method='drop', **kwargs):
        super(LinkingCleaning, self).__init__()
        self._map = linking.copy()
        accepted_methods = ['drop']
        assert method in accepted_methods
        self._method = method

    def filter_engine(self):
        n_entities = self._map.i.nunique()
        print(f'{self.__class__.__name__}: {n_entities} mapped entities found')

        duplicates = set(self._map.i[self._map.i.duplicated()].values)
        if self._method == 'drop':
            self._map = self._map[~self._map.i.isin(duplicates)]

        new_n_entities = self._map.i.nunique()
        entities_removed = n_entities - new_n_entities
        print(f'{self.__class__.__name__}: {entities_removed} entities removed')
        print(f'{self.__class__.__name__}: {new_n_entities} mapped entities retained')
        self._flag = entities_removed == 0

    def filter_output(self):
        return {'linking': self._map}


class KGTrainAlignment(Filter):

    def __init__(self, data, kg: pd.DataFrame, **kwargs):
        super(KGTrainAlignment, self).__init__()

        # loading
        self._dataset = data.copy()
        self._kg = kg.copy()

    def filter_engine(self):
        items = set(self._dataset.i)
        print(f'{self.__class__.__name__}: training set with {len(items)} items')
        subjects = set(self._kg.s)
        to_remove = set.difference(subjects, items)
        self._kg = self._kg[~self._kg.s.isin(to_remove)]
        print(f'{self.__class__.__name__}: entities removed from knowledge graph: {len(to_remove)}')
        self._flag = len(to_remove) == 0

    def filter_output(self):
        return {'data': self._dataset,
                'kg': self._kg}


class RemoveNoisyTriples(Filter):

    def __init__(self, kg: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        self._kg = kg.copy()
        self.noisy_preds = [
            'http://dbpedia.org/ontology/wikiPageWikiLink',
            'http://www.w3.org/2002/07/owl#sameAs',
            'http://schema.org/sameAs',
            'http://purl.org/linguistics/gold/hypernym',
            'http://www.w3.org/2000/01/rdf-schema#seeAlso',
            'http://dbpedia.org/property/wordnet_type',
            'http://dbpedia.org/ontology/wikiPageExternalLink',
            'http://dbpedia.org/ontology/thumbnail',
            'http://www.w3.org/ns/prov#wasDerivedFrom',
            'http://dbpedia.org/property/wikiPageUsesTemplate',
            'http://www.w3.org/2002/07/owl#differentFrom',
            'http://www.w3.org/1999/02/22-rdf-syntax-ns#type']

    def filter_engine(self):
        noisy_triples = self._kg.p.isin(self.noisy_preds)
        print(f'{self.__class__.__name__}: {sum(noisy_triples)} noisy triples found')
        self._kg = self._kg[~noisy_triples]
        self._flag = True

    def filter_output(self):
        return {'kg': self._kg}
