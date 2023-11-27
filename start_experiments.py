from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--config', type=str, default='movielens_best_kgtore')
parser.add_argument('--mail', action='store_true')
args = parser.parse_args()

if args.mail:
    from email_notifier.email_sender import EmailNotifier

    notifier = EmailNotifier()
    notifier.notify(run_experiment,
                    f"config_files/{args.config}.yml",
                    additional_body=f"config_files/{args.config}.yml")
else:
    run_experiment(f"config_files/{args.config}.yml")
