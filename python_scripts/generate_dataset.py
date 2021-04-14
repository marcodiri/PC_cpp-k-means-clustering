#!/usr/bin/python3

import numpy as np
from sklearn.datasets import load_digits, make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from python_scripts.utils import save_data


if __name__ == '__main__':
    np.random.seed(42)

    # digits dataset 1797 points
    # X_digits, y_digits = load_digits(return_X_y=True)
    # data = scale(X_digits)
    #
    # n_samples, n_features = data.shape
    # n_digits = len(np.unique(y_digits))
    # labels = y_digits
    #
    # reduced_data = PCA(n_components=2).fit_transform(data)
    # save_data(reduced_data, 'data.txt')

    # custom dataset 100k points
    dataset = make_blobs(n_samples=10**5, n_features=2, centers=10, return_centers=True)
    save_data(dataset[0], 'big_data.txt')
    save_data(dataset[1], 'big_data_labels.txt')
    save_data(dataset[2], 'big_data_centers.txt')
