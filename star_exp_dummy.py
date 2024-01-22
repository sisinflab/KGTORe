from elliot.run import run_experiment

files = ["config_files/facebook_best_kguf.yml", "config_files/movielens_best_kguf.yml"]
# run_experiment(f"config_files/yahoo_seed_expl.yml")
for file in files:
    run_experiment(file)