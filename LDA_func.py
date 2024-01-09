import numpy as np
from scipy import linalg

def mean_vec(data, labels):
    """Compute means vector for each class
    """
    class_labels = np.unique(labels)
    mu = np.empty((class_labels.shape[0], data.shape[1]))

    for i in class_labels:
        # print(i)
        idx_c = np.where(labels==i)[0]
        # print(idx_c)
        mu[i] = data[labels==i].mean(axis=0, keepdims=True)
    return mu
def var_vec(data, labels):
    """Compute means vector for each class
    """
    class_labels = np.unique(labels)
    var = np.empty((len(class_labels)))
    for i in class_labels:
        var[i] = np.var(data[labels==i])
    return var

def scatter_within(data, labels):
    """Compute within classes scatter matrix:
    S_w = sum_{class_i}S_i
    S_i = cov matrix (scaled by n)
    """
    class_labels = np.unique(labels)
    n_coeff = data.shape[1]
    mu_classes = mean_vec(data, labels)
    S_w = np.zeros((n_coeff, n_coeff))
    for i in class_labels:
        norm_Xi = data[labels==i] - mu_classes[i]
        for row in norm_Xi:
            row = np.expand_dims(row, 1)
            S_w += row@row.T
    return S_w/data.shape[0]
def scatter_between(data, labels):
    class_labels = np.unique(labels)
    n_coeff = data.shape[1]
    mu_classes = mean_vec(data, labels)
    mu_total = data.mean(axis=0, keepdims=True)
    S_b = np.zeros((n_coeff, n_coeff))
    for i in class_labels:
        mean_diff = mu_classes[i]-mu_total
        S_b += (mean_diff.T@mean_diff)*(data[labels==i].shape[0])
    return S_b/data.shape[0]

def project_vector(Sw, Sb, n_components=1):
    eig_vals, eig_vecs = linalg.eigh(Sb, Sw)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    V = np.vstack([eig_pairs[i][1] for i in range(0, n_components)]).reshape(Sw.shape[0], n_components)
    return V
