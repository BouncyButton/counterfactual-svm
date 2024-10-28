import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def binarize_labels(y):
    return y % 2


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def compute_diversity(cfs):
    if isinstance(cfs, torch.Tensor):
        cfs = cfs.cpu().numpy()
    cfs = np.array(cfs)
    C = cfs.shape[0]
    diversity = 0
    for i in range(C - 1):
        for j in range(i + 1, C):
            diversity += np.linalg.norm(cfs[i] - cfs[j], ord=2)

    return 1 / C ** 2 * diversity


def compute_ddp(cfs, submethod="inverse_dist"):
    """Computes the DPP of a matrix."""
    import torch
    compute_dist = lambda x, y: torch.norm(x - y, p=2)

    if not isinstance(cfs, torch.Tensor):
        cfs = torch.tensor(cfs)
    total_CFs = cfs.shape[0]
    det_entries = torch.ones((total_CFs, total_CFs))
    if submethod == "inverse_dist":
        for i in range(total_CFs):
            for j in range(total_CFs):
                det_entries[(i, j)] = 1.0 / (1.0 + compute_dist(cfs[i], cfs[j]))
                if i == j:
                    det_entries[(i, j)] += 0.0001

    elif submethod == "exponential_dist":
        for i in range(total_CFs):
            for j in range(total_CFs):
                det_entries[(i, j)] = 1.0 / (torch.exp(compute_dist(cfs[i], cfs[j])))
                if i == j:
                    det_entries[(i, j)] += 0.0001

    diversity_loss = torch.det(det_entries)
    return diversity_loss.item()
