from elliot.run import run_experiment
from config_kgtore import KGTORE_CONFIG
import argparse
import os

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, nargs='+', default=['facebook_book'])
parser.add_argument('--npr', type=str, nargs='+', default=[10, 30, 40])
parser.add_argument('--layer', type=str, nargs='+', default=[3, 2])
parser.add_argument('--alpha', type=str, nargs='+', default=[0.2, 0.4, 0.6, 0.8])
parser.add_argument('--beta', type=str, nargs='+', default=[0.2, 0.4, 0.6, 0.8])
parser.add_argument('--criterion', type=str, nargs='+', default=['entropy', 'gini'])
args = parser.parse_args()

nprs = args.npr
criterion = args.criterion
layers = args.layer
dataset = args.dataset
alphas = args.alpha
betas = args.beta

print('Exploration of parameters:\n'
      f'npr: {nprs}\n'
      f'criterion: {criterion}\n'
      f'layer: {layers}\n'
      f'dataset: {dataset}\n'
      f'alpha : {alphas}\n'
      f'beta: {betas}\n')

for d in dataset:
    for c in criterion:
        for npr in nprs:
            for layer in layers:
                for alpha in alphas:
                    for beta in betas:
                        print(f'Starting training with '
                              f'dataset: {d}\n'
                              f'criterion: {c}\n'
                              f'npr: {npr}\n'
                              f'layer: {layer}\n'
                              f'alpha: {alpha}\n'
                              f'beta: {beta}')

                        config = KGTORE_CONFIG.format(dataset=d,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      layers=layer,
                                                      npr=npr,
                                                      strategy=c)
                        config_dir = './config_files'
                        config_path = os.path.join(config_dir, 'runtime_conf.yml')
                        with open(config_path, 'w') as file:
                            file.write(config)
                        run_experiment(config_path)

                        print(f'Starting training with '
                              f'dataset: {d}\n'
                              f'criterion: {c}\n'
                              f'npr: {npr}\n'
                              f'layer: {layer}\n'
                              f'alpha: {alpha}\n'
                              f'beta: {beta}')
