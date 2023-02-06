from tqdm import tqdm
import numpy as np
import torch
import os


from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .KGTOREModel import KGTOREModel
from .DecisionPaths import DecisionPaths
from .LoadEdgeFeatures import LoadEdgeFeatures


class KGTORE(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._sampler = cs.Sampler(self._data.i_train_dict)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_lr", "lr", "lr", 0.0005, float, None),
            ("_elr", "elr", "elr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_alpha", "alpha", "alpha", 0.5, float, None),
            ("_beta", "beta", "beta", 0.5, float, None),
            ("_l_ind", "l_ind", "l_ind", 0.5, float, None),
            ("_ind_edges", "ind_edges", "ind_edges", 0.01, float, None),
            ("_n_layers", "n_layers", "n_layers", 1, int, None),
            ("_npr", "npr", "npr", 10, int, None),
            ("_criterion", "criterion", "criterion", "entropy", str, None),
            ("_loader", "loader", "loader", "KGTORETSVLoader", None, None)
        ]

        self.autoset_params()
        self._side = getattr(self._data.side_information, self._loader, None)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        row, col = data.sp_i_train.nonzero()

        try:
            name = 'decision_path' + str(self._npr) + "_" + str(self._criterion) + ".tsv"
            item_features_name = 'item_features' + str(self._npr) + "_" + str(self._criterion) + ".pk"
            dataset_path = os.path.abspath(os.path.join('./data', config.dataset, 'kgtore', name))
            item_features_path = os.path.abspath(os.path.join('./data', config.dataset, 'kgtore', item_features_name))
            print(f'Looking for {dataset_path}')
            print(f'Looking for {item_features_path}')
            self.edge_features, self.item_features = LoadEdgeFeatures(dataset_path, item_features_path, self._data.transactions)
            print("loaded edge features from: ", dataset_path, '\n')
        except:
            u_values, u_indices = np.unique(row, return_index=True)
            u_indices = np.append(u_indices, len(col))
            u_i_ordered_dict = {u_values[i]: col[u_indices[i]:u_indices[i + 1]] for i in range(len(u_values))}
            Dec_Paths_class = DecisionPaths(interactions=data.i_train_dict,
                                            u_i_dict=u_i_ordered_dict,
                                            kg=self._side.feature_map,
                                            public_items=data.public_items,
                                            public_users=data.public_users,
                                            transaction=self._data.transactions,
                                            device=device,
                                            df_name=config.dataset,
                                            criterion=self._criterion,
                                            npr=self._npr
                                            )
            self.edge_features = Dec_Paths_class.edge_features
            self.item_features = Dec_Paths_class.item_features

        col = [c + self._num_users for c in col]
        self.edge_index = np.array([list(row) + col, col + list(row)])
        self.num_interactions = row.shape[0]

        print(f'Number of KGTORE features: {self.edge_features.size(1)}')

        self._alpha = 1 - self._alpha
        self._beta = 1 - self._beta

        self._model = KGTOREModel(
            num_users=self._num_users,
            num_items=self._num_items,
            num_interactions=self.num_interactions,
            learning_rate=self._lr,
            edges_lr=self._elr,
            embedding_size=self._factors,
            l_w=self._l_w,
            alpha=self._alpha,
            beta=self._beta,
            l_ind=self._l_ind,
            ind_edges=self._ind_edges,
            n_layers=self._n_layers,
            edge_index=self.edge_index,
            edge_features=self.edge_features,
            item_features=self.item_features,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "KGTORE" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

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

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(gu[offset: offset_stop], gi)
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
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
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
