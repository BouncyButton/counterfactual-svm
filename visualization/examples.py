from draft.october.method.counterfactuals_v2 import Methods
from draft.october.run import cached_run_all

df = cached_run_all(datasets=['mnist'],
                    method_list=['wachter', 'proposal', 'dice'],
                    num_iter_list=[100],
                    n_counterfactuals_list=[1],
                    n_clusters_list=[32],
                    clustering_method_list=['kmedoids'],
                    cv_grid_size=5,
                    max_samples=3000,
                    strategy_list=['greedy'],
                    beta_list=[10.], n_jobs=2)

print(df)

columns = [0, 1, 2, 4, 6]
cv_fold = 0
wachter = Methods.get('wachter')
proposal = Methods.get('proposal')
dice = Methods.get('dice')

m_wachter = df[(df['Method'] == wachter) & (df['CV_index'] == cv_fold)]
m_proposal = df[(df['Method'] == proposal) & (df['CV_index'] == cv_fold)]
m_dice = df[(df['Method'] == dice) & (df['CV_index'] == cv_fold)]

import matplotlib.pyplot as plt

# set font size
plt.rcParams.update({'font.size': 18})
# tight layout
fig = plt.figure(figsize=(len(columns) * 5 / 2.2, 3 * 5 / 2))
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)
plt.subplots_adjust(wspace=0, hspace=0)
rows = ['source', wachter, dice, proposal]
for i, method in enumerate(rows):
    for j, idx in enumerate(columns):
        ax = fig.add_subplot(len(
            rows), len(columns), i * len(columns) + j + 1)
        # remove ax ticks
        if method == 'source':
            img = df[(df['Method'] == wachter)
                     & (df['CV_index'] == cv_fold)
                     & (df['Test_index'] == idx)].iloc[0]
            ax.imshow(img['Original_Data'].reshape(28, 28))
        else:
            row = df[(df['Method'] == method)
                     & (df['CV_index'] == cv_fold)
                     & (df['Test_index'] == idx)].iloc[0]
            img = row['Counterfactual']
            ax.imshow(img.reshape(28, 28))
        # if i == 0:
        #    ax.set_title(f"Example {idx}")
        if j == 0:
            ax.set_ylabel(method)
        ax.set_xticks([])
        ax.set_yticks([])
# plt.show()
import os
outdir = '../output/examples'
if not os.path.exists(outdir):
    os.makedirs(outdir)
plt.savefig(os.path.join(outdir, 'mnist_examples.pdf'))