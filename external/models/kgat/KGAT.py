from tqdm import tqdm
import torch
import os
import pandas as pd
import dgl
import math
from operator import itemgetter

from elliot.utils.write import store_recommendation
from .custom_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .KGATModel import KGATModel

from ast import literal_eval as make_tuple


class KGAT(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_kg_factors", "kg_factors", "kg_factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_aggr", "aggr", "aggr", 'gcn', str, None),
            ("_weight_size", "weight_size", "weight_size", "(64,32,16)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_message_dropout", "message_dropout", "message_dropout", 0.1, float, None),
            ("_loader", "loader", "loader", "KGINTSVLoader", None, None)
        ]

        self.autoset_params()
        self._side = getattr(self._data.side_information, self._loader, None)
        self.public_entities = {**self._data.public_items,
                                **self._side.public_objects}
        self.private_entities = {**self._data.private_items, **self._side.private_objects}
        self.items = list(self._data.public_items.values())
        mapped_subjects = list(itemgetter(*self._side.map_['subject'].tolist())(self.public_entities))
        mapped_objects = list(itemgetter(*self._side.map_['object'].tolist())(self.public_entities))
        mapped_relations = list(itemgetter(*self._side.map_['predicate'].tolist())(self._side.public_relations))

        kg_graph = pd.concat([pd.Series(mapped_subjects), pd.Series(mapped_relations), pd.Series(mapped_objects)],
                             axis=1)
        kg_graph.columns = ['subject', 'predicate', 'object']

        self._sampler = Sampler(self._data.i_train_dict, kg_graph)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dgl_graph = dgl.graph((torch.tensor(mapped_subjects).to(device), torch.tensor(mapped_objects).to(device)))
        dgl_graph.edata['relation_id'] = torch.tensor(mapped_relations).to(device)

        self._model = KGATModel(
            num_users=self._num_users,
            num_entities=self._side.n_entities,
            num_relations=self._side.n_relations - 1,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            kg_embed_k=self._kg_factors,
            aggr=self._aggr,
            l_w=self._l_w,
            weight_size=self._weight_size,
            message_dropout=self._message_dropout,
            kg_graph=dgl_graph,
            rows=mapped_subjects,
            cols=mapped_objects,
            data=mapped_relations,
            random_seed=self._seed,
        )

    @property
    def name(self):
        return "KGAT" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch, batch_kg in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch, batch_kg)

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()
            self.evaluate(it, loss / (it + 1))
            self._model.update_attentive_A()

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(torch.tensor(list(range(offset, offset_stop)), dtype=torch.int64), self.items)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
