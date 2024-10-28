from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine, make_moons, fetch_openml
import numpy as np
import pandas as pd


class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.X, self.y = self.load()

    def load(self):
        if self.dataset_name == 'iris':
            return load_iris(return_X_y=True)
        elif self.dataset_name == 'digits':
            return load_digits(return_X_y=True)
        elif self.dataset_name == 'breast_cancer':
            return load_breast_cancer(return_X_y=True)
        elif self.dataset_name == 'wine':
            return load_wine(return_X_y=True)
        elif self.dataset_name == 'moons':
            return make_moons(n_samples=200, noise=0.2, random_state=44)
        elif self.dataset_name == 'mnist':
            mnist = fetch_openml('mnist_784')
            y = np.array([int(label) for label in mnist.target])
            X = np.array(mnist.data)
            return X, y
        elif self.dataset_name == 'boston':
            data_url = "http://lib.stat.cmu.edu/datasets/boston"
            raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
            X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
            y = raw_df.values[1::2, 2] >= 20
            return X, y

        elif self.dataset_name == 'magic':
            data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
            raw_df = pd.read_csv(data_url, header=None)
            X = raw_df.values[:, :-1]
            y = raw_df.values[:, -1] == 'g'
            return X, y

        elif self.dataset_name == 'banknote':
            data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
            raw_df = pd.read_csv(data_url, header=None)
            X = raw_df.values[:, :-1]
            y = raw_df.values[:, -1] == 1
            return X, y

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def __call__(self, *args, **kwargs):
        return self.X, self.y