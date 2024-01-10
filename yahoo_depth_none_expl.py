from elliot.run import run_experiment
from config_yahoo_depth_none import TEMPLATE
import os

CONFIG_DIR = './config_files'

assert os.path.exists(CONFIG_DIR)

# datasets = ['facebook_book', 'yahoo_movies', 'movielens']
# alphas = [str(i) for i in [0.4]]
a = str(0.4)
nprs = [str(i) for i in [1, 2, 5, 10, 20]]
dataset = 'yahoo_movies'

for npr in nprs:
    config = TEMPLATE.format(dataset, dataset, dataset, dataset, alpha=a,  npr=npr)
    name = dataset + '_iron_tore' + '_' + 'a' + a + '_' + npr + '_' 'None' + '.yml'
    config_path = os.path.join(CONFIG_DIR, name)
    with open(config_path, 'w') as file:
        file.write(config)
    run_experiment(config_path)
