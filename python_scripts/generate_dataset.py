#!/usr/bin/python3

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from python_scripts.utils import save_data


if __name__ == '__main__':
    np.random.seed(42)

    X_digits, y_digits = load_digits(return_X_y=True)
    data = scale(X_digits)

    n_samples, n_features = data.shape
    n_digits = len(np.unique(y_digits))
    labels = y_digits

    reduced_data = PCA(n_components=2).fit_transform(data)
    save_data(reduced_data, 'data.txt')
