import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.dataset.samplers import custom_sampler as cs

from .UserFeatureMapper import UserFeatureMapper
from .KGFlexModel import KGFlexModel


class KGFlex(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, *args, **kwargs):
        # auto parameters
        self._params_list = [
            ("_lr", "lr", "lr", 0.01, None, None),
            ("_embedding", "embedding", "em", 10, int, None),
            ("_max_features_per_user", "max_features_per_user", "uf", None, None, None),
            ("_loader", "loader", "load", "KGRec", None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._side = getattr(self._data.side_information, self._loader, None)
        self._sampler = cs.Sampler(self._data.i_train_dict)

        max_features_per_user = self._params.max_features_per_user
        embedding = self._embedding
        logger = self.logger
        learning_rate = self._lr

        item_features = {self._data.public_items[i]: v for i, v in self._side.item_features.items()}

        # ------------------------------ USER FEATURES ------------------------------
        logger.info('Features info: user features selection...')
        self.user_feature_mapper = UserFeatureMapper(data=self._data,
                                                     item_features=item_features,
                                                     max_features_per_user=max_features_per_user)

        # ------------------------------ MODEL FEATURES ------------------------------
        logger.info('Features info: features mapping...')
        features = set()
        user_features = self.user_feature_mapper.user_features
        for _, f in user_features.items():
            features = set.union(features, set(f))

        item_features_selected = {item: set.intersection(item_features.get(item, {}), features) for
                                  item in self._data.private_items}

        feature_key_mapping = dict(zip(features, range(len(features))))

        logger.info('Features info: {} features found'.format(len(features)))

        # ------------------------------ MODEL ------------------------------
        self._model = KGFlexModel(data=self._data,
                                  n_features=len(features),
                                  learning_rate=learning_rate,
                                  embedding_size=embedding,
                                  user_features=user_features,
                                  item_features=item_features_selected,
                                  feature_key_mapping=feature_key_mapping)

    @property
    def name(self):
        return "KGFlex" \
               + "_e:" + str(self._epochs) \
               + f"_{self.get_params_shortcut()}"

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_recs(u, mask, k) for u in tqdm(self._data.users)}

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)
        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()
            self.evaluate(it, loss)
