import os
from elliot.run import run_experiment

CONFIG_DIR = './config_files'

facebook_best_models = ['facebook_noknowledge_best',
                        'facebook_kahfm_best',
                        'facebook_kgflex_best',
                        'facebook_kgin_best',
                        'facebook_kgtore_best']

yahoo_best_models = ['yahoo_noknowledge_best',
                     'yahoo_kahfm_best',
                     'yahoo_kgflex_best',
                     'yahoo_kgin_best',
                     #'facebook_kgtore_best'
                     ]

movielens_best_models = ['movielens_noknowledge_best',
                         'movielens_kahfm_best',
                         'movielens_kgflex_best',
                         'movielens_kgin_best']

best_models = movielens_best_models

for b in best_models:
    path = os.path.join(CONFIG_DIR, b + '.yml')
    assert os.path.exists(path), f'{path} not found'

for b in best_models:
    run_experiment(os.path.join(CONFIG_DIR, b + '.yml'))
