import os

from matplotlib import pyplot as plt

from draft.october.run import cached_run_all

outdir = '../output/runtime-vs-performance'
if not os.path.exists(outdir):
    os.makedirs(outdir)


def run_view(dataset_name, ax, idx_row, idx_col, y_axis='Distance'):
    df = cached_run_all(datasets=[dataset_name],
                        method_list=['proposal', 'wachter', 'dice'],
                        num_iter_list=[1, 3, 10, 30, 100, 300, 1000],
                        n_counterfactuals_list=[1],
                        n_clusters_list=[-1],
                        clustering_method_list=['kmedoids'],
                        cv_grid_size=20,
                        max_samples=3000)

    # to compute the means, first group by method dataset and CV_index, compute the median; then group by CV_index, and compute the mean
    # first, filter out invalid cfs
    df_of_interest = df[(df['Invalid'] == 0) & (df['Dataset'] == dataset_name)]
    # df_of_interest = df[(df['Dataset'] == dataset_name)]
    if len(df_of_interest) == 0:
        return

    # replace Distance with 9999999 if it is invalid
    # df_of_interest[y_axis] = df_of_interest[y_axis].where(df_of_interest['Invalid'] == False, 9999999)

    aggregate_per_fold = df_of_interest.groupby(['Method', 'Dataset', 'CV_index', 'Num_iter']).mean(numeric_only=True)
    grouped = aggregate_per_fold.groupby(['Method', 'Dataset', 'Num_iter'])
    means = grouped.mean()
    stds = grouped.std()

    # move indexes as columns
    means.reset_index(inplace=True)
    stds.reset_index(inplace=True)

    # make a plot with x: runtime; y: distance L2
    # errbars for std (both directions)
    from draft.october.method.counterfactuals_v2 import Methods

    proposal = Methods['proposal']
    wachter = Methods['wachter']
    dice = Methods['dice']

    # show error bars for each method
    for method in [proposal, wachter, dice]:
        std = stds[stds['Method'] == method]
        xx = means[means['Method'] == method]['Computation_Time']
        yy = means[means['Method'] == method][y_axis]

        ax.errorbar(xx, yy, xerr=std['Computation_Time'], yerr=std[y_axis], fmt='', label=f'{method}')

        validities = 1 - df.groupby(['Method', 'Dataset', 'Num_iter']).mean(numeric_only=True)['Invalid']
        # filter method and dataset
        validities = validities[(validities.index.get_level_values('Method') == method) & (
                validities.index.get_level_values('Dataset') == dataset_name)]

        # scatter each point with a different size according to validity
        # for x, y, validity in zip(xx, yy, validities):
        #     if validity < 1:
        #         plt.scatter(x, y, s=100, alpha=0.5, c='white', edgecolors='black')
        #
        #     plt.scatter(x, y, s=validity * 100, alpha=0.5, c='green' if validity == 1 else 'red', edgecolors='green' if validity == 1 else 'red')
        #
        l = list(zip(range(len(xx)), validities))
        color = 'blue' if method == Methods['proposal'] else 'orange' if method == Methods['wachter'] else 'green'
        for i, label in [l[1], l[-1]]:
            # with padding
            ax.text(xx.iloc[i] * 2.8, yy.iloc[i] * 0.93, f'{label:.0%}', ha='right', va='bottom', fontsize=15,
                    color='black',
                    alpha=0.3,
                    bbox=dict(facecolor='white', alpha=0.7,
                              edgecolor=color, boxstyle='round,pad=0.5'),
                    rotation=0, clip_on=True)

    if idx_row == 0:
        ax.set_title(dataset_name)
    if idx_row == len(metrics) - 1:
        ax.set_xlabel('Computation Time')
    if idx_col == 0:
        if y_axis == 'Distance':
            ax.set_ylabel('Distance (lower is better)')
        elif y_axis == 'Density':
            ax.set_ylabel('Density (higher is better)')
    ax.set_xscale('log')


datasets = list(sorted(['iris', 'breast_cancer', 'boston', 'banknote', 'magic']))
metrics = ['Distance', 'Density']
# font siz
plt.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(len(datasets) * 5, len(metrics) * 5))
for i, y_axis in enumerate(metrics):
    for j, dataset in enumerate(datasets):
        ax = fig.add_subplot(len(metrics), len(datasets), i * len(datasets) + j + 1)
        run_view(dataset, ax, i, j, y_axis=y_axis)

# create more space for the legend
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(left=0.05)
plt.subplots_adjust(right=0.99)
plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2),
           fancybox=True, shadow=True, ncol=5)
plt.savefig(os.path.join(outdir, f'runtime-vs-performance.pdf'))
# plt.show()
