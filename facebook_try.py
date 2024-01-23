from elliot.run import run_experiment
from fb_template_try import TEMPLATE
import os

CONFIG_DIR = './config_files'

assert os.path.exists(CONFIG_DIR)

# datasets = ['facebook_book', 'yahoo_movies', 'movielens']
alphas = [str(i) for i in [0.4, 0.6]]
# a = str(0.6)
# depths = [str(i) for i in [1, 2, 5, 10, 15, 20]]
# nprs = [str(i) for i in [1, 2, 5, 10, 20]]
depths = [str(None)]
nprs = [str(i) for i in [1]]
dataset = 'facebook_book'
for a in alphas:
    for npr in nprs:
        for depth in depths:
            config = TEMPLATE.format(dataset, dataset, dataset, dataset, alpha=a, depth=depth, npr=npr)
            name = dataset + '_iron_tore' + '_' + 'a' + a + '_' + npr + '_' + depth + '.yml'
            config_path = os.path.join(CONFIG_DIR, name)
            with open(config_path, 'w') as file:
                file.write(config)
            run_experiment(config_path)
