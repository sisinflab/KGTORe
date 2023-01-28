import os
from elliot.run import run_experiment

CONFIG_DIR = './config_files'

best_models = ['facebook_best_models',
               'facebook_kahfm_best',
               'facebook_kgflex_best',
               'facebook_kgin_best',
               'facebook_kgtore_best']

for b in best_models:
    assert os.path.exists(os.path.join(CONFIG_DIR, b + '.yml'))

for b in best_models:
    run_experiment(os.path.join(CONFIG_DIR, b + '.yml'))
