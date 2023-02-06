from os import path
from elliot.run import run_experiment
from data_preprocessing import movielens_preprocessing, facebook_book_preprocessing, yahoo_movies_preprocessing

movielens_data_folder = './data/movielens'
facebook_book_folder = './data/facebook_book'
yahoo_movies_folder = './data/yahoo_movies'

# PRE-PROCESSING

facebook_book_preprocessing.run(data_folder=facebook_book_folder)
yahoo_movies_preprocessing.run(data_folder=yahoo_movies_folder)
movielens_preprocessing.run(data_folder=movielens_data_folder)

# RUN BASELINES

# Facebook Books configuration files
facebook_baseline_configs = [
    'config_files/facebook_cf.yml',
    'config_files/facebook_kgflex.yml',
    'config_files/facebook_kahfm.yml',
    'config_files/facebook_kgin.yml'
]

facebook_kgtore_config = [
    'config_files/facebook_kgtore.yml'
]

# Yahoo Movies configuration files
yahoo_baseline_configs = [
    'config_files/yahoo_cf.yml',
    'config_files/yahoo_kgflex.yml',
    'config_files/yahoo_kahfm.yml',
    'config_files/yahoo_kgin.yml'
]

yahoo_kgtore_config = [
    'config_files/yahoo_kgtore.yml'
]

# MovieLens 1M configuration files
movielens_baseline_configs = [
    'config_files/movielens_cf.yml',
    'config_files/movielens_kgflex.yml',
    'config_files/movielens_kahfm.yml',
    'config_files/movielens_kgin.yml'
]

movielens_kgtore_config = [
    'config_files/movielens_kgtore.yml'
]

# check that all the configs exists
for config in facebook_baseline_configs + facebook_kgtore_config + yahoo_baseline_configs + yahoo_kgtore_config + movielens_baseline_configs + movielens_kgtore_config:
    assert path.exists(config)

# run the experiments for Facebook Books
for config in facebook_baseline_configs + facebook_kgtore_config:
    run_experiment(config)

# run the experiments for Yahoo Movies
for config in yahoo_baseline_configs + yahoo_kgtore_config:
    run_experiment(config)

# run the experiments for MovieLens 1M
for config in movielens_baseline_configs + movielens_kgtore_config:
    run_experiment(config)
