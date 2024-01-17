from elliot.run import run_experiment
from config_yahoo_seed import TEMPLATE
import os

CONFIG_DIR = './config_files'

assert os.path.exists(CONFIG_DIR)

# datasets = ['facebook_book', 'yahoo_movies', 'movielens']
# alphas = [str(i) for i in [0.4]]
a = str(0.4)
depths = [str(i) for i in [1, 2, 5, 10, 15, 20, None]]
nprs = [str(i) for i in [1, 2, 5, 10, 20]]
seeds = [str(i) for i in [229, 364, 291, 485, 237, 266]]


dataset = 'yahoo_movies'
for npr in nprs:
    for depth in depths:
        for seed in seeds:
            config = TEMPLATE.format(dataset, dataset, dataset, dataset, alpha=a, depth=depth, npr=npr, seed=seed)
            name = dataset + '_iron_tore' + '_' + 'a' + a + '_' + npr + '_' + depth + '_' + seed + '.yml'
            config_path = os.path.join(CONFIG_DIR, name)
            with open(config_path, 'w') as file:
                file.write(config)
            run_experiment(config_path)
