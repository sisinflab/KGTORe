from elliot.run import run_experiment
from config_kgtore_ab_exploration import KGTORE_CONFIG
import argparse
import os

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, nargs='+', default=['yahoo_movies'])
parser.add_argument('--npr', type=str, nargs='+', default=[40])
parser.add_argument('--layer', type=str, nargs='+', default=[3])
parser.add_argument('--alpha', type=str, nargs='+', default=[0.2, 0.4, 0.6, 0.8])
parser.add_argument('--beta', type=str, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
parser.add_argument('--gamma', type=float, default=0.5)
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
gamma = args.gamma

print('Exploration of parameters:\n'
      f'npr: {nprs}\n'
      f'criterion: {criterion}\n'
      f'layer: {layers}\n'
      f'dataset: {dataset}\n'
      f'alpha : {alphas}\n'
      f'beta: {betas}\n'
      f'gamma: {gamma}\n'
      )

for d in dataset:
    for alpha in alphas:
        for beta in betas:
            for npr in nprs:

                assert d in ['facebook_book', 'movielens', 'yahoo_movies']

                ind_edges = {'facebook_book': 0.01,
                             'movielens': 0.00001,
                             'yahoo_movies': 0.001}

                lrs = {'facebook_book': 0.00041155060113429626,
                       'yahoo_movies': 0.000664750387531533,
                       'movielens': 0.00020519696043968285}
                elrs = {'facebook_book': 0.00041155060113429626,
                        'yahoo_movies': 0.00305727041704464,
                        'movielens': 0.0014346784702323088}
                l_ws = {'facebook_book': 0.00041155060113429626,
                        'yahoo_movies': 0.00312734853526998,
                        'movielens': 0.060707796238838284}
                gammas = {'facebook_book': 0.00041155060113429626,
                          'yahoo_movies': 0.054524737,
                          'movielens': 0.0043878992507734}

                batch_sizes = {'facebook_book': 64,
                               'yahoo_movies': 256,
                               'movielens': 2048}

                lr = float(lrs[d])
                elr = float(elrs[d])
                l_w = float(l_ws[d])
                gamma = float(gamma)
                batch = int(batch_sizes[d])

                print(f'Starting training with '
                      f'dataset: {d}\n'
                      f'alpha: {alpha}\n'
                      f'beta: {beta}\n'
                      f'lr: {lr}\n'
                      f'elr: {elr}\n'
                      f'l_w: {l_w}\n'
                      f'gamma: {gamma}\n'
                      f'batch: {batch}\n'
                      )

                if int(float(beta)) == 1:
                    gamma = 0

                config = KGTORE_CONFIG.format(dataset=d,
                                              alpha=float(alpha),
                                              beta=float(beta),
                                              lr=lr,
                                              elr=elr,
                                              l_w=l_w,
                                              gamma=gamma,
                                              gpu=int(gpu),
                                              ind_edges=float(ind_edges[d]),
                                              batch=batch)

                config_dir = './config_files'
                config_path = os.path.join(config_dir, 'runtime_conf.yml')
                with open(config_path, 'w') as file:
                    file.write(config)
                run_experiment(config_path)

                print(f'Starting training with '
                      f'dataset: {d}\n'
                      f'alpha: {alpha}\n'
                      f'beta: {beta}')
