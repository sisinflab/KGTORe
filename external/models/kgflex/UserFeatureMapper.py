import random
import math
import multiprocessing as mp
# mp.set_start_method('fork')
from tqdm import tqdm
from itertools import islice
from operator import itemgetter
from collections import OrderedDict, Counter


def worker(user, user_items, neg_items, item_features, limit, seed):
    random.seed(seed)
    counter = user_features_counter(user_items, neg_items, item_features)
    return user, limited_selection(counter, limit)


class UserFeatureMapper:
    def __init__(self, data, item_features: dict, item_features2=None, random_seed=42,
                 max_features_per_user=100, second_order_limit=0):
        # for all the users compute the information for each feature
        self._data = data
        self._item_features = item_features
        # self._item_features2 = item_features2
        self._max_features_per_user = max_features_per_user
        # self._second_order_limit = second_order_limit
        self._users = self._data.private_users.keys()
        self._items = set(self._data.private_items.keys())
        # self._depth = 2

        self.user_features = self.user_features_selected_mp()

    def user_features_selected_mp(self):
        def args():
            return ((u,
                     set(self._data.i_train_dict[u].keys()),
                     set.difference(self._items, set(self._data.i_train_dict[u].keys())),
                     self._item_features,
                     self._max_features_per_user,
                     random.randint(0, 100000)) for u in self._users)

        arguments = args()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(worker, tqdm(arguments, desc='user features selection', total=len(self._users)))
        return {u: f for u, f in results}

    def user_features_selected(self, user):
        user_items = set(self._data.i_train_dict[user].keys())
        neg_items = set.difference(self._items, user_items)
        # counters = dict()
        counter = user_features_counter(user_items, neg_items, self._item_features)
        # counters[2] = user_features_counter(user_items, neg_items, self._item_features2)

        return user, limited_selection(counter, self._max_features_per_user)


def user_features_counter(user_items, neg_items, item_features_):
    def count_features():
        """
        Given a list of positive and negative items retrieves all them features and then counts them.
        :return: Counter, Counter
        """

        pos_features = []
        for p in positives:
            pos_features.extend(item_features_.get(p, set()))

        neg_features = []
        for n in negatives:
            neg_features.extend(item_features_.get(n, set()))

        pos_counter = Counter(pos_features)
        neg_counter = Counter(neg_features)

        return pos_counter, neg_counter

    # for each postive item pick a negative
    negatives = random.choices(list(neg_items), k=len(user_items))
    positives = user_items

    # count positive feature and negative features
    pos_c, neg_c = count_features()
    return pos_c, neg_c, len(positives)


def features_entropy(pos_counter, neg_counter, counter):
    """
    :param pos_counter: number of times in which feature is true and target is true
    :param neg_counter: number of times in which feature is true and target is false
    :param counter: number of items from which features have been extracted
    :return: dictionary feature: entropy with descending order by entropy
    """

    def relative_gain(partial, total):
        if total == 0:
            return 0
        ratio = partial / total
        if ratio == 0:
            return 0
        return - ratio * math.log2(ratio)

    def info_gain(pos_c, neg_c, n_items):

        den_1 = pos_c + neg_c

        h_pos = relative_gain(pos_c, den_1) + relative_gain(neg_c, den_1)
        den_2 = 2 * n_items - (pos_c + neg_c)

        num_1 = n_items - pos_c
        num_2 = n_items - neg_c
        h_neg = relative_gain(num_1, den_2) + relative_gain(num_2, den_2)

        return 1 - den_1 / (den_1 + den_2) * h_pos - den_2 / (den_1 + den_2) * h_neg

    attribute_entropies = dict()
    for positive_feature in pos_counter:
        ig = info_gain(pos_counter[positive_feature], neg_counter[positive_feature], counter)
        if ig > 0:
            attribute_entropies[positive_feature] = ig

    return OrderedDict(sorted(attribute_entropies.items(), key=itemgetter(1, 0), reverse=True))


def limited_selection(counter, limit):
    # 1st-order-features
    pos, neg, count = counter
    # 2nd-order-features
    # pos_2, neg_2, counter_2 = counters[2]

    if limit:
        assert limit > 0, "The number of semantic features must be positive"
        entropies = features_entropy(pos, neg, count)

        # top 10 1st-order-features ordered by entropy
        entropies_red = OrderedDict(islice(entropies.items(), limit))
        # filtering pos and neg features respect to the 'top limit' selected
        pos_f = Counter({k: pos[k] for k in entropies_red.keys()})
        neg_f = Counter({k: neg[k] for k in entropies_red.keys()})
    else:
        pos_f = pos
        neg_f = neg

    return features_entropy(pos_f, neg_f, count)
