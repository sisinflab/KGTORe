import os
from elliot.run import run_experiment

CONFIG_DIR = './config_files'

best_models = [
    'cf_facebook_best',
    'kahfm_facebook_results',
    'kgflex_facebook_results',
    'kgin_facebook_results'
    ]

for b in best_models:
    assert os.path.exists(os.path.join(CONFIG_DIR, b + '.yml'))

for b in best_models:
    run_experiment(os.path.join(CONFIG_DIR, b + '.yml'))
