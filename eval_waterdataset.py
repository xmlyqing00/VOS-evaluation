import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from WaterDataset.evaluation import Evaluation

default_water_path = '/Ship01/Dataset/water'
default_results_path = '/Ship01/Dataset/water/results'

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=False, default=default_water_path,
                    help='Path to the water folder.',)
parser.add_argument('--task', type=str, default='semi-supervised', choices=['semi-supervised', 'unsupervised'],
                    help='Task to evaluate the results', )
parser.add_argument('--method', type=str, required=True,
                    help='Method name.')
parser.add_argument('--update', action='store_true',
                    help='Recompute the performance results.' )
args, _ = parser.parse_known_args()
csv_name_global = 'global_results.csv'
csv_name_per_sequence = 'per-sequence_results.csv'
method_results_path = os.path.join(default_results_path, args.method)
print('Evaluate', args.method)

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join(method_results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(method_results_path, csv_name_per_sequence)
if not args.update and os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    print(f'Evaluating sequences for the {args.task} task...')
    # Create dataset and evaluate
    dataset_eval = Evaluation(root_folder=args.path, task=args.task)
    metrics_res = dataset_eval.evaluate(method_results_path)
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
sys.stdout.write(f"------------------------ Global results for {args.method} ------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.method} ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
print('')
