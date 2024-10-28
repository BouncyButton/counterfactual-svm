# -*- coding: utf-8 -*-
import argparse
from itertools import product

import numpy as np
import time

import pandas as pd
import torch
from joblib import Parallel, delayed

from experiment.exp1 import CounterfactualExperiment1


def _run_exp_by_idx(idx, experiments):
    return experiments[idx].run()

def cached_run_all(datasets=None,
                   method_list=None, num_iter_list=None, n_counterfactuals_list=None,
                   n_clusters_list=None, strategy_list=None, beta_list=None, lambda_list=None,
                   clustering_method_list=None, cv_grid_size=20, max_samples=3000, n_jobs=1):
    if method_list is None:
        method_list = ['wachter', 'proposal', 'dice']
    if num_iter_list is None:
        num_iter_list = [100]
    if n_counterfactuals_list is None:
        n_counterfactuals_list = [1]
    if n_clusters_list is None:
        n_clusters_list = [-1]
    if strategy_list is None:
        strategy_list = ['greedy']
    if beta_list is None:
        beta_list = [10.]
    if lambda_list is None:
        lambda_list = [10.]
    if clustering_method_list is None:
        clustering_method_list = ['kmedoids']
    if datasets is None:
        datasets = ['iris', 'breast_cancer']

    all_result = []
    experiments = []

    for dataset in datasets:
        for method, num_iter, n_counterfactuals, n_clusters, clustering_method, strategy, beta in list(product(
                method_list, num_iter_list, n_counterfactuals_list, n_clusters_list, clustering_method_list,
                strategy_list, beta_list)):
            experiment = CounterfactualExperiment1(
                dataset_name=dataset,
                n_clusters=n_clusters if method == 'proposal' else -1,
                cv_grid_size=cv_grid_size,
                num_iter=num_iter,
                max_samples=max_samples,
                method=method,
                n_counterfactuals=n_counterfactuals,
                clustering_method=clustering_method if method == 'proposal' and n_clusters > 0 else None,
                strategy=strategy,
                beta=beta,
            )
            experiments.append(experiment)

    # use joblib to parallelize the experiments
    results = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_run_exp_by_idx)(idx, experiments) for idx in range(len(experiments)))

    return pd.concat(results)

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # create parser
    parser = argparse.ArgumentParser(description='Run counterfactual experiments')

    parser.add_argument('--datasets', nargs='+', type=str, default=['iris'],
                        # ['iris', 'breast_cancer', 'wine', 'boston', 'magic', 'banknote', 'digits'],
                        help='Dataset to use')
    parser.add_argument('--methods', nargs='+', type=str, default=['wachter', 'dice', 'proposal'], help='Method to use')
    parser.add_argument('--num_iters', nargs='+', type=int, default=[10], help='Number of iterations')
    parser.add_argument('--n_clusters', nargs='+', type=int, default=[32],
                        help='Number of clusters, ignored by all expect proposal')
    parser.add_argument('--cv_grid_size', type=int, default=5, help='CV grid size')
    parser.add_argument('--max_samples', type=int, default=3000, help='Max samples for the train+test set')
    parser.add_argument('--n_counterfactuals', nargs='+', type=int, default=[1],
                        help='Number of counterfactuals to generate')
    parser.add_argument('--clustering_method', nargs='+', type=str, default=['kmedoids'],
                        help='Clustering method to use')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')

    args = parser.parse_args()

    datasets = args.datasets
    cv_grid_size = args.cv_grid_size
    method_list = args.methods
    num_iter_list = args.num_iters
    n_counterfactuals_list = args.n_counterfactuals
    n_clusters_list = args.n_clusters
    clustering_method_list = args.clustering_method

    start = time.time()
    all_result = cached_run_all(datasets=datasets, method_list=method_list, num_iter_list=num_iter_list,
                                n_counterfactuals_list=n_counterfactuals_list, n_clusters_list=n_clusters_list,
                                clustering_method_list=clustering_method_list, cv_grid_size=cv_grid_size,
                                max_samples=args.max_samples, n_jobs=args.n_jobs)
    print(all_result.mean(numeric_only=True))
    print(f'Elapsed time: {time.time() - start}')
