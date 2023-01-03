"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np


class Sampler:
    def __init__(self, indexed_ratings, kg, seed=42):
        np.random.seed(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._kg = kg
        self._lkg_dict = {s: len(self._kg[self._kg['subject'] == s]) for s in self._kg['subject'].unique().tolist()}
        self._n_heads = self._kg['subject'].nunique()
        self._n_tails = self._kg['object'].nunique()
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict
        lkg_dict = self._lkg_dict
        n_heads = self._n_heads
        n_tails = self._n_tails
        kg = self._kg

        def sample():
            u = r_int(n_users)
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            i = ui[r_int(lui)]

            j = r_int(n_items)
            while j in ui:
                j = r_int(n_items)

            try:
                s = r_int(n_heads)
                all_object_s = kg[kg['subject'] == s]['object'].tolist()
                p_s = kg[kg['subject'] == s]['predicate'].tolist()[r_int(lkg_dict[s])]
                o_i = kg[(kg['subject'] == s) & (kg['predicate'] == p_s)]['object'].values[0]
                o_j = r_int(n_tails)
                while o_j in all_object_s:
                    o_j = r_int(n_tails)
            except:
                print()

            return u, i, j, s, p_s, o_i, o_j

        for batch_start in range(0, events, batch_size):
            bui, bii, bij, h, p, tp, tn = map(np.array, zip(*[sample() for _ in range(batch_start, min(batch_start + batch_size, events))]))
            yield (bui[:, None], bii[:, None], bij[:, None]), (h[:, None], p[:, None], tp[:, None], tn[:, None])
