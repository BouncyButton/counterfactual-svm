import math
import os

import numpy as np

from draft.october.method.counterfactuals_v2 import Methods
from draft.october.run import cached_run_all

outdir = '../output/avg-rank'
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

    # df_with_invalid = df.groupby(['Dataset', 'Method', x_axis]).mean(numeric_only=False)

    # df = df[df['Invalid'] == 0]

    # compute the mean value of distance for each row in df
    df['Mean distance'] = df['Distance'].apply(lambda x: np.mean(x) if x is not None else None)
    df['Mean density'] = df['Density'].apply(lambda x: np.mean(x) if x is not None else None)

    grouped = df.groupby(['Dataset', x_axis, 'Test_index', 'CV_index'])

    # Step 2: Rank 'distance' across 'method' within each group
    df['mean distance rank'] = grouped['Mean distance'].transform(
        lambda x: x.rank(na_option='bottom', method='average', ascending=True))
    df['mean density rank'] = grouped['Mean density'].transform(
        lambda x: x.rank(na_option='bottom', method='average', ascending=True))
    df['mean diversity rank'] = grouped['Diversity'].transform(
        lambda x: x.rank(na_option='bottom', method='average', ascending=False))

    # Step 3: Compute average rank across 'Test_index' and 'CV_index' for each 'Dataset', 'x_axis', 'method'
    avg_rank_df = df.groupby(['Dataset', x_axis, 'Method']).agg(Mean_distance_rank=('mean distance rank', 'mean'),
                                                                Mean_density_rank=('mean density rank', 'mean'),
                                                                Diversity_rank=('mean diversity rank', 'mean')
                                                                ).reset_index()
    # add the new columns
    df = df.merge(avg_rank_df, on=['Dataset', x_axis, 'Method'])

    df.reset_index(inplace=True)

    df = df.groupby(['Dataset', 'Method', x_axis]).mean(numeric_only=False)
    # df_stds = df.groupby(['Dataset', 'Method', x_axis]).std(numeric_only=False)
    # create a grid of plots, one for each dataset

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 10))

    df.reset_index(inplace=True)
    n_rows = len(datasets)
    n_cols = len(metrics_considered)

    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(metrics_considered):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
            for method in method_list:
                method = Methods.get(method)
                x = df[(df['Dataset'] == dataset) & (df['Method'] == method)][x_axis]
                if metric == 'Invalid':
                    y = 1 - df[(df['Dataset'] == dataset) & (df['Method'] == method)]['Invalid']
                else:
                    y = df[(df['Dataset'] == dataset) & (df['Method'] == method)][metric]
                ax.plot(x, y, 'o-', label=f'{method}')
                # ax.legend()
            if i == 0:
                ax.set_title('Validity' if metric == 'Invalid' else metric)
            if i == len(datasets) - 1:
                ax.set_xlabel(x_axis)
            if j == 0:
                ax.set_ylabel(dataset)
    # put the legend outside
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{outdir}/{datasets}-fix2.pdf')


run_view(['iris', 'wine', 'breast_cancer', 'boston', 'banknote'],
         [1, 2], ['proposal', 'wachter', 'dice'],
         metrics_considered=['Invalid', 'Mean distance', 'Mean density', 'Diversity', 'Mean_distance_rank', 'Mean_density_rank', 'Diversity_rank'],
         x_axis='N_counterfactuals')
