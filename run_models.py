from os import path
from elliot.run import run_experiment
from data_preprocessing import movielens_preprocessing, facebook_book_preprocessing, yahoo_movies_preprocessing
from config_yahoo_template import TEMPLATE as yahoo_template
import os

yahoo_movies_folder = './data/yahoo_movies'
CONFIG_DIR = './config_files'
assert os.path.exists(CONFIG_DIR)

# Preprocessing
yahoo_movies_preprocessing.run(data_folder=yahoo_movies_folder)

# alpha beta parameters:
alphas = [str(i) for i in [0, 0.2, 0.4, 0.6, 0.8, 1]]
betas = [str(i) for i in [0, 0.2, 0.4, 0.6, 0.8, 1]]
# model type
models_aggr = ['zero', 'last']


dataset = 'yahoo_movies'
for aggr in models_aggr:
    for a in alphas:
        for b in betas:
            config = yahoo_template.format(dataset, dataset, dataset, dataset, alpha=a, beta=b, aggr=aggr)
            name = dataset + '_kgtore_' + aggr + '_' + 'a_' + a + 'b_' + b + '.yml'
            config_path = os.path.join(CONFIG_DIR, name)
            with open(config_path, 'w') as file:
                file.write(config)
            run_experiment(config_path)
