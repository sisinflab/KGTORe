from types import SimpleNamespace
import typing as t
import numpy as np
import pandas as pd

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
# forse vanno rimappati user e item dopo che sono stati mappati da dataset in questa classe


class KGINTSVLoader(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.attribute_file = getattr(ns, "kg", None)
        self.users = users
        self.items = items
        if self.attribute_file is not None:
            self.map_ = self.read_triplets(self.attribute_file)  # KG
        self.entities = set(self.map_.values[:, 0]).union(set(self.map_.values[:, 2]))  # soggetti + oggetti
        self.items = set.intersection(self.items, self.entities)  # inutile, il prefiltering già fatto
        self.entity_list = set.difference(self.entities, self.items)  # entities che non sono items

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items
        self.map_ = self.map_[self.map_['subject'].isin(self.items)]
        self.entities = set(self.map_.values[:, 0]).union(set(self.map_.values[:, 2]))
        self.items = set.intersection(self.items, self.entities)
        self.entity_list = set.difference(self.entities, self.items)

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "KGINTSVLoader"
        ns.object = self
        ns.__dict__.update(self.__dict__)
        ns.feature_map = self.map_   # è il kg
        ns.relations = np.unique(ns.feature_map.values[:, 1])
        ns.n_relations = len(ns.relations) + 1  # per la relazione di like
        ns.n_entities = len(self.items) + len(ns.entity_list)
        ns.n_nodes = ns.n_entities + len(self.users)
        ns.private_relations = {p[0] + 1: f for p, f in list(np.ndenumerate(ns.relations))} # npden = lista con indice valore: [ ((0,), 1),  ((1,), 7)]
        ns.public_relations = {v: k for k, v in ns.private_relations.items()}
        ns.private_objects = {p + len(self.items): f for p, f in list(enumerate(ns.entity_list))}  #solo per entità != item
        ns.public_objects = {v: k for k, v in ns.private_objects.items()}
        return ns

    def read_triplets(self, file_name):
        return pd.read_csv(file_name, sep='\t', header=None, names=['subject', 'predicate', 'object'])
        # return np.array(kg)
