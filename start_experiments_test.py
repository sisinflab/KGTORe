from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='movielens')
args = parser.parse_args()

run_experiment(f"config_files/kgtore_{args.dataset}_test.yml")
