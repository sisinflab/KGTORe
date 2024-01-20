from elliot.run import run_experiment
from config_yahoo_abl import TEMPLATE
import os

CONFIG_DIR = './config_files'

assert os.path.exists(CONFIG_DIR)

# datasets = ['facebook_book', 'yahoo_movies', 'movielens']
# alphas = [str(i) for i in [0.4]]
a = str(0.4)
nprs = [str(i) for i in [1]]
seeds = [str(i) for i in [229, 364, 291, 485, 237]]
# ablation_types = ["random", "shuffled"]
ablation_types = ["nofilter"]

dataset = 'yahoo_movies'
for abl in ablation_types:
    for npr in nprs:
        for seed in seeds:
            config = TEMPLATE.format(dataset, dataset, dataset, dataset, alpha=a, npr=npr, abl=abl, seed=seed)
            name = dataset + '_iron_tore' + '_' + 'a' + a + '_' + npr + '_' + abl + '_' + seed + '.yml'
            config_path = os.path.join(CONFIG_DIR, name)
            with open(config_path, 'w') as file:
                file.write(config)
            run_experiment(config_path)
