import os
from elliot.run import run_experiment

CONFIG_DIR = './config_files'

facebook_best_models = ['facebook_best_models',
                        'facebook_kahfm_best',
                        'facebook_kgflex_best',
                        'facebook_kgin_best',
                        'facebook_kgtore_best']

yahoo_best_models = ['yahoo_best_models',
                     'yahoo_kahfm_best',
                     'yahoo_kgflex_best',
                     'yahoo_kgin_best',
                     #'facebook_kgtore_best'
                     ]

movielens_best_models = ['movielens_best_models',
                         'movielens_kahfm_best',
                         'movielens_kgflex_best',
                         'movielens_kgin_best']

best_models = yahoo_best_models

for b in best_models:
    assert os.path.exists(os.path.join(CONFIG_DIR, b + '.yml'))

for b in best_models:
    run_experiment(os.path.join(CONFIG_DIR, b + '.yml'))
