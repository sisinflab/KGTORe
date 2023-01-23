from elliot.run import run_experiment
from config_metrics_template import METRICS_TEMPLATE
import argparse
import os

RECS_FOLDER = os.path.abspath('./results/recs')
CONFIG_DIR = './config_files'

assert os.path.exists(RECS_FOLDER)
assert os.path.exists(CONFIG_DIR)

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, nargs='+', default=['facebook_book'])
args = parser.parse_args()

datasets = args.dataset

for dataset in datasets:
    print(f'Computing recs for {dataset}')
    recs_folder = os.path.join(RECS_FOLDER, dataset)
    assert os.path.exists(recs_folder)

    #recs_paths = [os.path.join(recs_folder, p) for p in os.listdir(recs_folder) if p.endswith('.tsv')]

    #for recs_path in recs_paths:
    #    config = METRICS_TEMPLATE.format(dataset=dataset,
    #                                     recs=recs_path)

    config = METRICS_TEMPLATE.format(dataset=dataset, recs=recs_folder)
    config_path = os.path.join(CONFIG_DIR, 'runtime_metrics_conf.yml')
    with open(config_path, 'w') as file:
        file.write(config)
    run_experiment(config_path)
