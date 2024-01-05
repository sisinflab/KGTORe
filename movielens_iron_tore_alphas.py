from elliot.run import run_experiment
from config_movielens_iron_tore_template import TEMPLATE
import os

CONFIG_DIR = './config_files'

assert os.path.exists(CONFIG_DIR)

# datasets = ['facebook_book', 'yahoo_movies', 'movielens']
alphas = [str(i) for i in [0, 0.2, 0.4, 0.6, 0.8, 1]]
dataset = 'movielens'
for a in alphas:
    config = TEMPLATE.format(dataset, dataset, dataset, dataset, alpha=a)
    name = dataset + '_iron_tore' + '_' + a + '.yml'
    config_path = os.path.join(CONFIG_DIR, name)
    with open(config_path, 'w') as file:
        file.write(config)
    run_experiment(config_path)