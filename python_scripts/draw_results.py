#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from python_scripts.utils import load_data, DATA_PATH


if __name__ == '__main__':
    data = load_data("big_data.txt")
    centroids = load_data("kmeans_centroids.txt")
    labels = load_data("kmeans_labels.txt", int)

    groups = [[] for _ in np.unique(labels)]
    for i in range(len(data)):
        groups[labels[i]].append(data[i])

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    plt.figure(1)
    plt.clf()
    if len(groups) > 10:
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, len(groups))]
        plt.gca().set_prop_cycle('color', colors)
    for g in groups:
        plt.plot(np.array(g)[:, 0], np.array(g)[:, 1], '.', markersize=2)
    # Plot the centroids as a red X
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=60, linewidths=2,
                color='r', zorder=10)

    plt.title('K-means clustering\n'
              'Centroids are marked with red cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(DATA_PATH+"clusters_plot.png")
