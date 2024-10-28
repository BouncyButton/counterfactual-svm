import math
import os
from itertools import product

import numpy as np
import pandas as pd

from draft.october.method.counterfactuals_v2 import Methods
from draft.october.run import cached_run_all

outdir = '../output/table'
if not os.path.exists(outdir):
    os.makedirs(outdir)


# Define the ranking function
def rank_with_nan_handling(x):
    # Fill NaN with the smallest rank (i.e., highest numerical rank)
    return x.rank(na_option='bottom', method='average')


def run_view(datasets, n_counterfactuals_list, method_list, metrics_considered, x_axis='N_counterfactuals', cv_grid_size=20, n_clusters=-1):
    df = cached_run_all(datasets=datasets,
                        method_list=method_list,
                        num_iter_list=[100],
                        n_counterfactuals_list=n_counterfactuals_list,
                        n_clusters_list=[n_clusters],
                        clustering_method_list=['kmedoids'],
                        cv_grid_size=cv_grid_size,
                        max_samples=3000,
                        strategy_list=['greedy'],
                        beta_list=[10.], n_jobs=2)

    # i want to create a table with
    #        | method1 | method2 | method3 |
    # dataset| m1 m2 m3| m1 m2 m3| m1 m2 m3|
    #   d1   | 1  2  3 | 1  2  3 | 1  2  3 |
    #   d2   | 1  2  3 | 1  2  3 | 1  2  3 |

    # first i have to save the validity
    validity_grouped = df[['Dataset', 'Method', 'Invalid', 'CV_index']].groupby(
        ['Dataset', 'Method', 'CV_index'])
    validity_k_fold = validity_grouped.mean()
    validity_mean = 1 - validity_k_fold.groupby(['Dataset', 'Method']).mean()
    validity_std = validity_k_fold.groupby(['Dataset', 'Method']).std()

    df_with_no_invalid = df[df['Invalid'] == 0]
    df_with_no_invalid_grouped = df_with_no_invalid[['Dataset', 'Method', 'CV_index'] + metrics_considered].groupby(
        ['Dataset', 'Method', 'CV_index'])
    df_with_no_invalid_kfold = df_with_no_invalid_grouped.mean()
    df_with_no_invalid_mean = df_with_no_invalid_kfold.groupby(['Dataset', 'Method']).mean()
    df_with_no_invalid_std = df_with_no_invalid_kfold.groupby(['Dataset', 'Method']).std()

    table = []
    for dataset, method, metric in product(datasets, method_list, metrics_considered):
        m = Methods.get(method)
        if metric == 'Invalid':
            row = (
                f"{validity_mean.loc[(dataset, m)].values[0]:.0%} ± {validity_std.loc[(dataset, m)].values[0]:.0%}")
        else:
            row = (
                f"{df_with_no_invalid_mean.loc[(dataset, m), metric]:.2f} ± {df_with_no_invalid_std.loc[(dataset, m), metric]:.2f}")
        table.append({(dataset, m, metric): row})

    # Flatten the list of dictionaries and convert to a DataFrame
    flattened_data = {key: value for d in table for key, value in d.items()}
    df = pd.DataFrame.from_dict(flattened_data, orient='index', columns=['Value'])

    # Set the multi-level index
    df.index = pd.MultiIndex.from_tuples(df.index, names=['Dataset', 'Method', 'Metric'])

    # Reset index if needed
    df = df.reset_index()
    df_pivot = df.pivot_table(index='Dataset', columns=['Method', 'Metric'], values='Value', aggfunc='first')
    df_pivot = df_pivot[[Methods.get('wachter'), Methods.get('dice'), Methods.get('proposal')]]
    # Convert to LaTeX format and adjust formatting for each cell
    latex_table = df_pivot.to_latex(
        column_format="|c|" + "c" * len(df_pivot.columns) + "|",  # Column format for LaTeX
        header=True,
        index=True,
        multicolumn_format='c'
    )

    return latex_table


res = run_view(['iris', 'wine', 'breast_cancer', 'digits', 'boston', 'banknote', 'magic'],
               [1], ['wachter', 'dice', 'proposal'],
               metrics_considered=['Invalid', 'Density'])

print(res)

res = run_view(['iris', 'wine', 'breast_cancer', 'digits', 'boston', 'banknote', 'magic'],
               [1], ['wachter', 'dice', 'proposal'],
               metrics_considered=['Distance', 'Distance_L1'])

print(res)

res = run_view(['mnist'],
               [1], ['wachter', 'dice', 'proposal'],
               metrics_considered=['Distance', 'Distance_L1'], cv_grid_size=5, n_clusters=32)

print(res)
res = run_view(['mnist'],
               [1], ['wachter', 'dice', 'proposal'],
               metrics_considered=['Invalid', 'Density'], cv_grid_size=5, n_clusters=32)

print(res)
