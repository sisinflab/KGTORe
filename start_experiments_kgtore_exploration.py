from elliot.run import run_experiment
from config_kgtore_ab_exploration import KGTORE_CONFIG
import argparse
import os

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, nargs='+', default=['yahoo_movies'])
parser.add_argument('--npr', type=str, nargs='+', default=[40])
parser.add_argument('--layer', type=str, nargs='+', default=[3])
parser.add_argument('--alpha', type=str, nargs='+', default=[0.2, 0.4, 0.6, 0.8])
parser.add_argument('--beta', type=str, nargs='+', default=[0, 0.1, 0.3, 0.5, 0.7, 0.9])
parser.add_argument('--gamma', type=str, nargs='+', default=[0.5])
parser.add_argument('--criterion', type=str, nargs='+', default=['entropy'])
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

nprs = args.npr
criterion = args.criterion
layers = args.layer
dataset = args.dataset
alphas = args.alpha
betas = args.beta
gpu = args.gpu
gammas = args.gamma

print('Exploration of parameters:\n'
      f'npr: {nprs}\n'
      f'criterion: {criterion}\n'
      f'layer: {layers}\n'
      f'dataset: {dataset}\n'
      f'alpha : {alphas}\n'
      f'beta: {betas}\n'
      f'gamma: {gammas}\n'
      )

for d in dataset:
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                print(f'Starting training with '
                      f'dataset: {d}\n'
                      f'alpha: {alpha}\n'
                      f'beta: {beta}'
                      f'gamma: {gammas}\n'
                      )

                assert d in ['facebook_book', 'movielens', 'yahoo_movies']

                ind_edges = {'facebook_book': 0.01,
                               'movielens': 0.001,
                               'yahoo_movies': 0.001}

                config = KGTORE_CONFIG.format(dataset=d,
                                              alpha=float(alpha),
                                              beta=float(beta),
                                              gamma=float(gamma),
                                              gpu=int(gpu),
                                              ind_edges=float(ind_edges[d]))
                config_dir = './config_files'
                config_path = os.path.join(config_dir, 'runtime_conf.yml')
                with open(config_path, 'w') as file:
                    file.write(config)
                run_experiment(config_path)

                print(f'Starting training with '
                      f'dataset: {d}\n'
                      f'alpha: {alpha}\n'
                      f'beta: {beta}')
