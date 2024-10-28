import os

from sklearn.utils import shuffle
import pandas as pd
from tqdm import tqdm

from data.dataset import DatasetLoader
from utils.utils import binarize_labels, compute_diversity, standardize_data
from method.counterfactuals_v2 import CounterfactualWachter, CounterfactualProposal, CounterfactualDiCE
from sklearn.svm import SVC
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import time
from joblib import Memory

# Get the current file's directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# go up one level
current_file_dir = os.path.dirname(current_file_dir)

# Construct the path to the cachedir folder
cachedir = os.path.join(current_file_dir, "cachedir")

memory = Memory(cachedir, verbose=0)


@memory.cache
def memorized_train_model(X_train, y_train, cv_grid_size=20, random_state=42, n_folds=5):
    model = GridSearchCV(
        estimator=SVC(kernel='rbf', probability=True, random_state=random_state),
        param_grid={'C': np.logspace(-1, 3, cv_grid_size), 'gamma': np.logspace(-6, 0, cv_grid_size)},
        n_jobs=2, verbose=3, cv=n_folds
    )
    model.fit(X_train, y_train)
    return model


@memory.cache
def memorized_kde(X, step, n_folds=5):
    kde_cv = GridSearchCV(
        estimator=KernelDensity(),
        param_grid={'bandwidth': np.arange(0.1, 10.0, step)}, n_jobs=-1, cv=n_folds,
        verbose=1
    )
    kde_cv.fit(X)
    return KernelDensity(bandwidth=kde_cv.best_params_["bandwidth"]).fit(X)


def compute_cf_and_runtime(cf_method, x_orig, y_target, N=1, **kwargs):
    start_time = time.perf_counter()
    xcf = cf_method.compute_counterfactual(x_orig, target=2 * y_target - 1, N=N, **kwargs)
    end_time = time.perf_counter()
    return xcf, end_time - start_time


@memory.cache
def memorized_run(dataset_name=None,
                  n_folds=5,
                  max_samples=10000,
                  cv_grid_size=20,
                  method='proposal',
                  num_iter=100,
                  n_counterfactuals=1,
                  clustering_method='kmedoids',
                  n_clusters=-1,
                  random_state=42,
                  lambda_=10.,
                  beta=10.,
                  strategy='greedy'
                  ):
    X, y = DatasetLoader(dataset_name)()
    X, y = shuffle(X, y, random_state=random_state)
    X, y = X[:max_samples], y[:max_samples]
    y = binarize_labels(y)

    kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)

    results = []
    for cv_index, (train_index, test_index) in tqdm(list(enumerate(kf.split(X))),
                                                    desc=f'running kfold on {list(locals().items())[:12]}',
                                                    delay=60, leave=False):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        X_train, X_test = standardize_data(X_train, X_test)

        res = run_single_split(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                               cv_index=cv_index, cv_grid_size=cv_grid_size,
                               method=method, num_iter=num_iter, n_clusters=n_clusters,
                               n_counterfactuals=n_counterfactuals, clustering_method=clustering_method,
                               lambda_=lambda_, beta=beta, n_folds=n_folds, dataset_name=dataset_name,
                               max_samples=max_samples)
        results.extend(res)

    return pd.DataFrame(results)


def run_single_split(X_train=None, y_train=None, X_test=None, y_test=None, cv_index=None, cv_grid_size=20,
                     method='proposal', num_iter=100, n_clusters=-1, n_counterfactuals=1, clustering_method='kmedoids',
                     lambda_=10., beta=10., n_folds=5, dataset_name=None, max_samples=3000, strategy='greedy'):
    # Train classifier and fit density estimators
    model = train_classifier(X_train, y_train, cv_grid_size=cv_grid_size, random_state=42, n_folds=n_folds)
    kernel_density_estimators = fit_density_estimators(X_train, y_train, cv_grid_size=cv_grid_size)
    cf_method = init_method(model, method=method, X_train=X_train, y_train=y_train, num_iter=num_iter,
                            n_clusters=n_clusters, clustering_method=clustering_method, lambda_=lambda_, beta=beta)

    return generate_counterfactuals(model=model, method=cf_method, kernel_density_estimators=kernel_density_estimators,
                                    X_test=X_test, y_test=y_test, cv_index=cv_index,
                                    y_test_target='opposite', n_counterfactuals=n_counterfactuals,
                                    dataset_name=dataset_name, cv_grid_size=cv_grid_size, num_iter=num_iter,
                                    max_samples=max_samples, strategy=strategy)


def train_classifier(X_train, y_train, cv_grid_size=20, random_state=42, n_folds=5):
    return memorized_train_model(X_train, y_train, cv_grid_size=cv_grid_size,
                                 random_state=random_state, n_folds=n_folds)


def fit_density_estimators(X_train, y_train, cv_grid_size=20):
    kernel_density_estimators = {}
    labels = np.unique(y_train)
    for label in labels:
        X_ = X_train[y_train == label]
        kde = optimize_kde(X_, cv_grid_size=cv_grid_size)
        kernel_density_estimators[label] = kde
    return kernel_density_estimators


def optimize_kde(X, cv_grid_size=20):
    step = 1 / cv_grid_size
    return memorized_kde(X, step)


def init_method(model=None, method='proposal', X_train=None, y_train=None, num_iter=100, n_clusters=-1,
                clustering_method='kmedoids', lambda_=10., beta=10.):
    if method == 'wachter':
        return CounterfactualWachter(
            clf=model,
            lambda_=lambda_, num_iter=num_iter,
        )

    if method == 'dice':
        return CounterfactualDiCE(
            clf=model,
            X_train=X_train,
            y_train=y_train, num_iter=num_iter
        )

    if method == 'proposal':
        return CounterfactualProposal(
            clf=model,
            beta=beta,
            X_train=X_train,
            n_clusters=n_clusters,
            num_iter=num_iter,
            cluster_method=clustering_method
        )

    raise ValueError(f"Unknown method: {method}")


def generate_counterfactuals(model=None, method=None,
                             kernel_density_estimators=None, X_test=None, y_test=None,
                             cv_index=None, y_test_target='opposite', n_counterfactuals=1,
                             dataset_name=None, cv_grid_size=20, num_iter=100, max_samples=10000, strategy='greedy'):
    predictions = model.predict(X_test)
    if y_test_target == 'opposite':
        y_test_target = (predictions + 1) % 2

    result = []
    for i in tqdm(list(range(X_test.shape[0])), delay=120,
                  desc=f'testing using kfold method={method}', leave=False):
        x_orig = X_test[i, :]
        y_orig = y_test[i]
        pred = predictions[i]
        y_target = y_test_target[i]
        correct_prediction = pred == y_orig

        kde = kernel_density_estimators[y_target]

        xcf, t = compute_cf_and_runtime(method, x_orig, y_target, N=n_counterfactuals, strategy=strategy)
        res = format_results(model=model, cf_method=method, x_orig=x_orig, y_orig=y_orig, y_target=y_target,
                             xcf=xcf, elapsed_time=t, kde=kde, cv_index=cv_index, x_idx=i,
                             correct_prediction=correct_prediction, n_counterfactuals=n_counterfactuals,
                             dataset_name=dataset_name, cv_grid_size=cv_grid_size, num_iter=num_iter,
                             max_samples=max_samples, strategy=strategy)
        result.append(res)

    return result


def format_results(model=None, cf_method=None, x_orig=None, y_orig=None, y_target=None, xcf=None,
                   elapsed_time=None, kde=None, cv_index=None, x_idx=None, correct_prediction=None,
                   n_counterfactuals=1,
                   dataset_name=None, cv_grid_size=20, num_iter=100, max_samples=10000, strategy='greedy'):
    if xcf is not None:
        xcf = np.array(xcf)

    if xcf is None:
        invalid = True
    elif n_counterfactuals == 1:
        invalid = model.predict(xcf.reshape(1, -1))[0] != y_target
    else:
        invalid = np.any(model.predict(xcf) != y_target)

    diversity = None
    if xcf is None:
        density = None
        computation_time = None
        distance = None
        distance_l1 = None
        counterfactual = None
    elif xcf is not None and n_counterfactuals == 1:
        density = kde.score_samples(xcf.reshape(1, -1))[0]
        computation_time = elapsed_time
        distance = np.linalg.norm(x_orig - xcf)
        distance_l1 = np.sum(np.abs(x_orig - xcf))
        counterfactual = xcf.flatten()
    else:
        density = np.array([kde.score_samples(xcf_.reshape(1, -1))[0] for xcf_ in xcf])
        computation_time = elapsed_time
        distance = np.array([np.linalg.norm(x_orig - xcf_) for xcf_ in xcf])
        distance_l1 = np.array([np.sum(np.abs(x_orig - xcf_)) for xcf_ in xcf])
        counterfactual = np.array(xcf.reshape(-1, xcf.shape[-1]))
        diversity = compute_diversity(counterfactual)

    return {
        'Dataset': dataset_name,
        'CV_grid_size': cv_grid_size,
        'Num_iter': num_iter,
        'Max_samples': max_samples,
        'N_clusters': cf_method.n_clusters if hasattr(cf_method, 'n_clusters') else -1,
        'Test_index': x_idx,
        'Method': cf_method.get_class_name(),
        'Density': density,
        'Original_Data': x_orig,
        'Original_Data_Label': y_orig,
        'Counterfactual': counterfactual,
        'Counterfactual_Target_Label': y_target,
        'Computation_Time': computation_time,
        'Distance': distance,
        'Distance_L1': distance_l1,
        'Invalid': invalid,
        'CV_index': cv_index,
        'Clustering_type': cf_method.cluster_method if hasattr(cf_method, 'cluster_method') else 'none',
        'Correct_Prediction': correct_prediction,
        'N_counterfactuals': n_counterfactuals,
        'Lambda': cf_method.lambda_ if hasattr(cf_method, 'lambda_') else None,
        'Beta': cf_method.beta if hasattr(cf_method, 'beta') else None,
        'Diversity': diversity,
        'Strategy': strategy if hasattr(cf_method, 'strategy') else None
    }


class CounterfactualExperiment1:
    def __init__(self, dataset_name=None,
                 cv_grid_size=20,
                 n_clusters=-1,
                 num_iter=100,
                 max_samples=10000,
                 method='proposal',
                 n_counterfactuals=1,
                 X_train=None,
                 y_train=None,
                 random_state=42,
                 lambda_=10.,
                 beta=10.,
                 strategy='greedy',
                 clustering_method='kmedoids'):
        self.strategy = strategy
        self.beta = beta
        self.lambda_ = lambda_
        self.clustering_method = clustering_method
        self.dataset_name = dataset_name
        self.cv_grid_size = cv_grid_size
        self.n_clusters = n_clusters
        self.num_iter = num_iter
        self.max_samples = max_samples
        self.method = method
        self.results = []
        self.n_counterfactuals = n_counterfactuals
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state

    def load_data(self):
        dl = DatasetLoader(self.dataset_name)
        X, y = dl()
        return shuffle(X, y, random_state=self.random_state)

    def compute_cf_and_runtime(self, cf_method, x_orig, y_target):
        start_time = time.perf_counter()
        N = self.n_counterfactuals
        xcf = cf_method.compute_counterfactual(x_orig, target=2 * y_target - 1, N=N)
        end_time = time.perf_counter()
        return xcf, end_time - start_time

    def run(self):
        """Execute in a memorized way. This will cache the results of the experiment and avoid re-running the same exp.
        Also, I want to avoid using a sqlite database since it's a mess to keep track of the db file.
        In this way I just "imagine" I can run the experiment every time, and it will be fast and efficient.
        But in practice it is cached."""
        return memorized_run(
            dataset_name=self.dataset_name,
            n_folds=5,
            max_samples=self.max_samples,
            cv_grid_size=self.cv_grid_size,
            method=self.method,
            num_iter=self.num_iter,
            n_counterfactuals=self.n_counterfactuals,
            clustering_method=self.clustering_method,  # self.kwargs.get('clustering_method', 'none'),
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            lambda_=self.lambda_,
            beta=self.beta,
            strategy=self.strategy
        )
