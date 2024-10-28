import math
import os

import numpy as np

from draft.october.method.counterfactuals_v2 import Methods
from draft.october.run import cached_run_all

outdir = '../output/betas'
if not os.path.exists(outdir):
    os.makedirs(outdir)


def run_view(datasets, n_counterfactuals_list, method_list, metrics_considered, beta_list, x_axis='Beta',
             z_axis='N_counterfactuals'):
    df = cached_run_all(datasets=datasets,
                        method_list=method_list,
                        num_iter_list=[100],
                        n_counterfactuals_list=n_counterfactuals_list,
                        n_clusters_list=[-1],
                        clustering_method_list=['kmedoids'],
                        cv_grid_size=20,
                        max_samples=3000,
                        strategy_list=['greedy'],
                        beta_list=beta_list)

    df_with_invalid = df.groupby(['Dataset', z_axis, x_axis]).mean(numeric_only=False)

    df = df[df['Invalid'] == 0]

    # compute the mean value of distance for each row in df
    df['Mean distance'] = df['Distance'].apply(lambda x: np.mean(x) if x is not None else None)
    df['Mean density'] = df['Density'].apply(lambda x: np.mean(x) if x is not None else None)
    df_fold = df.groupby(['Dataset', z_axis, x_axis, 'CV_index']).mean(numeric_only=False)
    df_fold = df_fold.reset_index()
    df = df_fold.groupby(['Dataset', z_axis, x_axis]).mean(numeric_only=True)
    df_std = df_fold.groupby(['Dataset', z_axis, x_axis]).std(numeric_only=True)
    df_std.reset_index(inplace=True)
    # create a grid of plots, one for each dataset

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(len(datasets) * 5, len(metrics_considered) * 5))
    # set font
    plt.rcParams.update({'font.size': 18})
    # remove space left and right
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.1)

    # remove n_counterfactuals from index but keep dataset and method
    df.reset_index(inplace=True)
    n_rows = len(metrics_considered)
    n_cols = len(datasets)

    z_axis_values = df[z_axis].unique()
    for i, metric in enumerate(metrics_considered):
        for j, dataset in enumerate(datasets):
            ax = fig.add_subplot(n_rows, n_cols, i * len(datasets) + j + 1)
            # log x
            ax.set_xscale('log')
            for z_value in z_axis_values:
                # this does not work anymore for method..
                x = df[(df['Dataset'] == dataset) & (df[z_axis] == z_value)][x_axis]
                std = df_std[(df_std['Dataset'] == dataset) & (df_std[z_axis] == z_value)][metric]
                if metric == 'Invalid':
                    y = 1 - df_with_invalid.loc[(dataset, z_value, x), 'Invalid']
                else:
                    y = df[(df['Dataset'] == dataset) & (df[z_axis] == z_value)][metric]
                ax.plot(x, y, 'o-', label=f'{z_value}')
                # ax.errorbar(x, y, yerr=std, fmt='o-', label=f'{z_value}')

            if i == 0:
                ax.set_title(dataset)
            if j == 0:
                y_label = 'Distance (lower is better)' if metric == 'Mean distance' else 'Density (higher is better)' if metric == 'Mean density' else 'Diversity (higher is better)' if metric == 'Diversity' else 'Validity (higher is better)' if metric == 'Invalid' else metric
                ax.set_ylabel(y_label)
            if i == n_rows - 1:
                ax.set_xlabel(x_axis)

    # put the legend outside
    plt.legend(loc='upper center', bbox_to_anchor=(0., -0.2),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig(f'{outdir}/{datasets}-{x_axis}-{z_axis}.pdf')


run_view(list(sorted(['iris', 'breast_cancer', 'boston', 'wine', 'magic'])),
         [1, 2, 3, 4], ['proposal'],
         metrics_considered=['Mean distance', 'Mean density', 'Diversity'],
         x_axis='Beta', z_axis='N_counterfactuals', beta_list=[0.1, 0.3, 1, 3, 10, 30, 100])
