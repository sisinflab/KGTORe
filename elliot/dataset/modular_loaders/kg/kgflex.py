from types import SimpleNamespace
import typing as t
from os.path import splitext
import ast

import numpy as np
import pandas as pd

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class KGFlexLoader(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.item_features_path = getattr(ns, "item_features", None)
        self.kg_path = getattr(ns, "kg_path", None)
        self.users = users
        self.items = items

        if self.item_features_path:
            self.item_features = self.item_features = pd.read_csv(self.item_features_path, header=None, sep='\t', names=['item', 'feature'], converters={"feature": ast.literal_eval})
            self.item_features = self.item_features.groupby('item')['feature'].apply(set).to_dict()
        else:
            if self.kg_path:
                kg = pd.read_csv(self.kg_path, sep='\t', names=['subject', 'predicate', 'object'],
                                 dtype={'subject': int, 'predicate': int, 'object': int})
                self.item_features = pd.DataFrame()
                self.item_features['item'] = kg['subject']
                self.item_features['feature'] = list(zip(kg['predicate'], kg['object']))
                self.item_features = self.item_features.groupby('item')['feature'].apply(set).to_dict()
            else:
                raise Exception("Side information not provided. Please load a file with item features or ...")

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users, items):
        self.users = self.users & users
        self.items = self.items & items
        self.item_features = {k: self.item_features[k] for k in self.items}

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "KGFlexLoader"
        ns.object = self
        ns.__dict__.update(self.__dict__)
        return ns

    @staticmethod
    def read_triples(path: str) -> t.List[t.Tuple[str, str, str]]:
        triples = []

        tmp = splitext(path)
        ext = tmp[1] if len(tmp) > 1 else None

        with open(path, 'rt') as f:
            for line in f.readlines():
                if ext is not None and ext.lower() == '.tsv':
                    s, p, o = line.split('\t')
                else:
                    s, p, o = line.split()
                triples += [(s.strip(), p.strip(), o.strip())]
        return triples
