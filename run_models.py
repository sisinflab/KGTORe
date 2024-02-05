from os import path
from elliot.run import run_experiment
import os

config_name = 'yahoo_kgtore_bef.yml'
config_path = os.path.join('config_files', config_name)
run_experiment(config_path)

config_name = 'yahoo_kgtore_last.yml'
config_path = os.path.join('config_files', config_name)
run_experiment(config_path)