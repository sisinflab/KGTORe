from elliot.run import run_experiment
from config_kgtore import KGTORE_CONFIG
import argparse
import os

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='facebook')
args = parser.parse_args()

nprs = [10, 30, 40]
criterion = ['entropy']
layers = [3, 2]
features_embedding_size = [4, 16]
dataset = ['facebook_book']

for d in dataset:
    for c in criterion:
        for npr in nprs:
            for layer in layers:
                for fe in features_embedding_size:
                    print(f'Starting training with '
                          f'dataset: {d}\n'
                          f'criterion: {c}\n'
                          f'npr: {npr}\n'
                          f'layer: {layer}\n'
                          f'feature embedding size: {fe}')

                    config = KGTORE_CONFIG.format(dataset=d,
                                                  features=fe,
                                                  layers=layer,
                                                  npr=npr,
                                                  strategy=c)
                    config_dir = './config_files'
                    config_path = os.path.join(config_dir, 'runtime_conf.yml')
                    with open(config_path, 'w') as file:
                        file.write(config)
                    run_experiment(config_path)

                    print(f'End training with'
                          f'dataset: {d}\n'
                          f'criterion: {c}\n'
                          f'npr: {npr}\n'
                          f'layer: {layer}\n'
                          f'feature embedding size: {fe}')
