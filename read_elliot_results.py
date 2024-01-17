import os
import pandas as pd
file_type = ".tsv"
PATH = './prova/yahoo_seed'

assert os.path.exists(PATH)

parameters = ['seed', 'depth', 'npr']
delimiter = '_'
metrics = ['nDCGRendle2020', 'HR', 'Precision', 'Recall', 'ItemCoverage', 'SEntropy', 'Gini']

directory = os.fsencode(PATH)

final_results = pd.DataFrame(columns=parameters + metrics)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    file_path = os.path.join(PATH, filename)

    if filename.endswith(file_type):
        df = pd.read_csv(file_path, sep='\t')
        model_parameter = df['model'][0]
        file_results = dict()
        for param in parameters:
            param_value = ''.join(model_parameter.split(param + '=')[1].split(delimiter)[0])
            param_value = param_value.replace("$", ".")
            param_value = [int(param_value)]
            file_results[param] = param_value
        for metric in metrics:
            file_results[metric] = df[metric][0]
        file_results = pd.DataFrame.from_dict(file_results)
        final_results = pd.concat([final_results, file_results])
    else:
        continue

print()