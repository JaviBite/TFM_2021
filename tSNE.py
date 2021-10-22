
    
from sklearn import decomposition
from sklearn.manifold import TSNE
import cv2
import numpy as np
import torch
import sys


def main():
    file1 = sys.argv[1]
    files = np.load(file1, allow_pickle=True)
    X, labels = files['a'], files['b']
    y = labels

    X = X.reshape(X.shape[0],X.shape[2])

    print(X.shape)
    print(labels.shape)

    pca = decomposition.PCA(n_components=50)
    pca.fit(X)
    Xpca = pca.transform(X)

    # RUN TSNE

    ############################################################
    # Fit and transform with a TSNE
    
    tsne = TSNE(n_components=2, random_state=0)
    # Just 2 components in order show to a 2D plot

    ############################################################
    # Project the data in 2D
    X_2d = tsne.fit_transform(Xpca)

    ############################################################
    # Visualize the data
    target_names = ['poner','remover']
    target_ids = range(len(target_names))

    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()