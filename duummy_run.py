from os import path
from elliot.run import run_experiment
import os

config_name = 'facebook_best_kgtore.yml'
# config_name = 'yahoo_movies_kgtore_ABL_zero_a_0b_0.yml'
config_path = os.path.join('config_files', config_name)
run_experiment(config_path)