from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='movielens')
args = parser.parse_args()

npr = [1, 3, 10]
criterion = ['entropy', 'gini']

for n in npr:
    for c in criterion:
        print(f'Starting training with npr: {n}, criterion: {c}')
        run_experiment(f"config_files/kgtore_{args.dataset}_{n}_{c}.yml")
        print(f'End training with npr: {n}, criterion: {c}')
