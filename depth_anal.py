import os
import pandas as pd
import numpy as np


path = './data/depth_analysis'
dataset_names = ['yahoo', 'facebook']
criterion = 'entropy'
nprs = [1, 2, 5, 10, 15, 20]

def create_path(start_dir: str, dataset_name: str, criterion: str, npr: int) -> str:
    df_name = 'depth' + str(npr) + '_' + criterion + '.tsv'
    path = os.path.abspath(os.path.join(start_dir, dataset_name,criterion, df_name))
    return path

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    return df

def depth_analysis(df_dict: dict, nprs: list) -> np.array:
    data = pd.DataFrame(np.array([[int(npr), df_dict[npr]['depth'].mean(), df_dict[npr]['depth'].std(), np.max(df_dict[npr]['depth']), np.min(df_dict[npr]['depth'])] for npr in nprs]))
    data.columns = ['npr', 'mean_depth', 'std_depth', 'max_depth', 'min_depth']
    data['npr'] = data['npr'].astype(int)
    return data



results = dict()
for dataset_name in dataset_names:
    df_dict = dict()
    for npr in nprs:
        current_path = create_path(start_dir=path, dataset_name=dataset_name, criterion=criterion, npr=npr)
        df_dict[npr] = load_dataset(current_path)
    results[dataset_name] = depth_analysis(df_dict=df_dict, nprs=nprs)

print()






