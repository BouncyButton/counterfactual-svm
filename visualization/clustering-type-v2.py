import math
import os

import numpy as np

from draft.october.method.counterfactuals_v2 import Methods
from draft.october.run import cached_run_all

outdir = '../output/clustering-type'
if not os.path.exists(outdir):
    os.makedirs(outdir)


def run_view(datasets, n_counterfactuals_list, method_list, metrics_considered, beta_list, x_axis='Beta',
             z_axis='N_counterfactuals'):
    df = cached_run_all(datasets=datasets,
                        method_list=method_list,
                        num_iter_list=[100],
                        n_counterfactuals_list=n_counterfactuals_list,
                        n_clusters_list=[-2, 64, 32, 16, 8, 4, 2, 1],
                        clustering_method_list=['kmedoids', 'kmeans'],
                        cv_grid_size=20,
                        max_samples=3000,
                        strategy_list=['greedy'],
                        beta_list=beta_list)

    df = df[df['Invalid'] == 0]
    # replace None in z_axis with 'None'
    df[z_axis] = df[z_axis].apply(lambda x: 'None' if x is None else x)
    grouped = df.groupby(['Dataset', z_axis, x_axis, 'CV_index']).mean(numeric_only=True).groupby(
        ['Dataset', z_axis, x_axis])
    df = grouped.mean()
    df_stds = grouped.std()

    # create a grid of plots, one for each dataset

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(len(metrics_considered) * 5, len(datasets) * 5))
    # set font
    plt.rcParams.update({'font.size': 18})
    # tight layout
    # plt.tight_layout()

    # remove n_counterfactuals from index but keep dataset and method
    df.reset_index(inplace=True)
    n_rows = len(datasets)
    n_cols = len(metrics_considered)

    z_axis_values = list(sorted(df[z_axis].unique()))
    if 'None' in z_axis_values:
        z_axis_values.remove('None')

    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics_considered):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
            # log x
            ax.set_xscale('log')
            for z, z_value in enumerate(z_axis_values):
                x = df[(df['Dataset'] == dataset) & (df[z_axis] == z_value) & (df[x_axis] > 0)][x_axis]
                y = df[(df['Dataset'] == dataset) & (df[z_axis] == z_value) & (df[x_axis] > 0)][metric]
                # ax.plot(x, y, 'o-', label=f'{z_value}')
                # add error bars
                std = df_stds[(df_stds.index.get_level_values('Dataset') == dataset) &
                              (df_stds.index.get_level_values(z_axis) == z_value) & (
                                          df_stds.index.get_level_values(x_axis) > 0)][metric]
                ax.errorbar(x, y, yerr=std, fmt='o-', label=f'{z_value}', alpha=0.5)
                # ax.legend()
            y_value_for_no_cluster = df[(df['Dataset'] == dataset) & (df[x_axis] == -1)][metric]
            if len(y_value_for_no_cluster) > 0:
                ax.axhline(y_value_for_no_cluster.values[0], color='r', linestyle='--', label='SV only')
            y_value_for_no_cluster = df[(df['Dataset'] == dataset) & (df[x_axis] == -2)][metric]
            if len(y_value_for_no_cluster) > 0:
                ax.axhline(y_value_for_no_cluster.values[0], color='b', linestyle='--', label='All')
            if i == 0:
                ax.set_title('Validity' if metric == 'Invalid' else metric)
            if i == len(datasets) - 1:
                ax.set_xlabel(x_axis)
            if j == 0:
                ax.set_ylabel(dataset)
    # make sure to have space on the bottom
    plt.subplots_adjust(bottom=0.25)

    # put the legend on the bottom
    plt.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.2),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig(f'{outdir}/{datasets}-{x_axis}-{z_axis}.pdf')


run_view(['digits'],
         [1], ['proposal'],
         beta_list=[10.],
         metrics_considered=['Distance', 'Density'],
         x_axis='N_clusters', z_axis='Clustering_type')
