import math
import os

import numpy as np

from draft.october.method.counterfactuals_v2 import Methods
from draft.october.run import cached_run_all

outdir = '../output/diversity'
if not os.path.exists(outdir):
    os.makedirs(outdir)


# Define the ranking function
def rank_with_nan_handling(x):
    # Fill NaN with the smallest rank (i.e., highest numerical rank)
    return x.rank(na_option='bottom', method='average')


def run_view(datasets, n_counterfactuals_list, method_list, metrics_considered, x_axis='N_counterfactuals'):
    df = cached_run_all(datasets=datasets,
                        method_list=method_list,
                        num_iter_list=[100],
                        n_counterfactuals_list=n_counterfactuals_list,
                        n_clusters_list=[-1],
                        clustering_method_list=['kmedoids'],
                        cv_grid_size=20,
                        max_samples=3000,
                        strategy_list=['greedy'],
                        beta_list=[10.])

    df_with_invalid = df.groupby(['Dataset', 'Method', x_axis, 'CV_index']).mean(numeric_only=True)
    df_with_invalid.reset_index(inplace=True)
    df_with_invalid_mean = df_with_invalid.groupby(['Dataset', 'Method', x_axis]).mean(numeric_only=True)
    df_with_invalid_mean.reset_index(inplace=True)
    df_with_invalid_std = df_with_invalid.groupby(['Dataset', 'Method', x_axis]).std(numeric_only=True)
    df_with_invalid_std.reset_index(inplace=True)

    df = df[df['Invalid'] == 0]

    # compute the mean value of distance for each row in df
    df['Mean distance'] = df['Distance'].apply(lambda x: np.mean(x) if x is not None else None)
    # df['Mean distance rank'] = df.groupby(['Dataset', 'Method', x_axis])['Mean distance'].transform(
    #     rank_with_nan_handling)
    df['Mean density'] = df['Density'].apply(lambda x: np.mean(x) if x is not None else None)
    # df['Mean density rank'] = df.groupby(['Dataset', 'Method', x_axis])['Mean density'].transform(
    #    rank_with_nan_handling)
    # df = df.groupby(['Dataset', 'Method', x_axis]).mean(numeric_only=False)
    # first
    df_fold = df.groupby(['Dataset', 'Method', x_axis, 'CV_index']).mean(numeric_only=False)
    df_fold = df_fold.reset_index()
    df = df_fold.groupby(['Dataset', 'Method', x_axis]).mean(numeric_only=True)
    df_std = df_fold.groupby(['Dataset', 'Method', x_axis]).std(numeric_only=True)
    df_std.reset_index(inplace=True)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(len(datasets) * 5, len(metrics_considered) * 5))
    # change space
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)

    # fontsize
    plt.rcParams.update({'font.size': 18})

    # remove n_counterfactuals from index but keep dataset and method
    df.reset_index(inplace=True)
    n_rows = len(metrics_considered)
    n_cols = len(datasets)

    for i, metric in enumerate(metrics_considered):
        for j, dataset in enumerate(datasets):
            ax = fig.add_subplot(n_rows, n_cols, i * len(datasets) + j + 1)
            # set xlim
            ax.set_xlim([min(n_counterfactuals_list) - 0.1, max(n_counterfactuals_list) + 0.1])
            for k, method in enumerate(method_list):
                method = Methods.get(method)
                x = df[(df['Dataset'] == dataset) & (df['Method'] == method)][x_axis] + k * 0.02
                yerr = df_std[(df_std['Dataset'] == dataset) & (df_std['Method'] == method)][metric]
                if metric == 'Invalid':
                    y = 1 - df_with_invalid_mean[
                        (df_with_invalid_mean['Dataset'] == dataset) & (df_with_invalid_mean['Method'] == method)][
                        metric]
                    yerr = df_with_invalid_std[
                        (df_with_invalid_std['Dataset'] == dataset) & (df_with_invalid_std['Method'] == method)][metric]
                else:
                    y = df[(df['Dataset'] == dataset) & (df['Method'] == method)][metric]
                # ax.plot(x, y, 'o-', label=f'{method}', alpha=0.7)
                ax.errorbar(x, y, yerr=yerr, fmt='o-', label=f'{method}', alpha=0.7)
                # ax.legend()
            if i == 0:
                ax.set_title(dataset)
            if i == len(metrics_considered) - 1:
                ax.set_xlabel(x_axis)
            if j == 0:
                y_label = 'Distance (lower is better)' if metric == 'Mean distance' else 'Density (higher is better)' if metric == 'Mean density' else 'Diversity (higher is better)' if metric == 'Diversity' else 'Validity (higher is better)' if metric == 'Invalid' else metric
                ax.set_ylabel(y_label)

    # put the legend outside
    plt.legend(loc='upper center', bbox_to_anchor=(0.11, -0.22),
               fancybox=True, shadow=True, ncol=5)
    plt.savefig(f'{outdir}/{datasets}-fix3.pdf')


run_view(['iris', 'breast_cancer', 'boston', 'wine', 'magic'],
         [4, 3, 2, 1], ['proposal', 'wachter', 'dice'],
         metrics_considered=['Invalid', 'Mean distance', 'Mean density', 'Diversity'],
         x_axis='N_counterfactuals')
