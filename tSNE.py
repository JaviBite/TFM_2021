
    
from sklearn import decomposition
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import numpy as np
import sys


def main():
    file1 = sys.argv[1]
    files = np.load(file1, allow_pickle=True)
    X, labels = files['a'], files['b']
    y = np.array(labels)

    metadata = ['poner','remover','voltear']

    N_CLASSES = np.max(labels) + 1
    N_SAMPLES = X.shape[0]
    N_TIMESTEPS = X.shape[1]

    HOG_H = X.shape[2]
    HOG_W = X.shape[3]
    ORIENTATIONS = X.shape[4]

    if N_TIMESTEPS == 1:
        X = X.reshape(N_SAMPLES,HOG_H*HOG_W*ORIENTATIONS)
    else:
        #X = X[:,1,:,:,:]
        #X = X.reshape(N_SAMPLES,HOG_H*HOG_W*ORIENTATIONS)
        
        #Xx = [X[i,1,:,:,:].ravel() for i in range(N_SAMPLES)]  
        #X = np.array(Xx)
        
        X = X.reshape(N_SAMPLES*N_TIMESTEPS,HOG_H*HOG_W*ORIENTATIONS) 
        y = [val for val in y for _ in range(N_TIMESTEPS)]  
        y = np.array(y) 

    print(X.shape)
    print(y.shape)

    pca = decomposition.PCA(n_components=50)
    pca.fit(X)
    Xpca = pca.transform(X)

    # RUN TSNE
    
    TSNE_DIM = 2
    if len(sys.argv) < 2:
        print("Error: Usage: [dataset_file.npz] [num_tsne 2 or 3]")
    elif len(sys.argv) == 2:
        TSNE_DIM = 2
    elif len(sys.argv) >= 3:
        TSNE_DIM = int(sys.argv[2])
        
    print("Running TSNE with dimensionality = ", TSNE_DIM)
    
    if TSNE_DIM == 2:

        ############################################################
        # Fit and transform with a TSNE
        
        tsne = TSNE(n_components=2, random_state=0)
        # Just 2 components in order show to a 2D plot

        ############################################################
        # Project the data in 2D
        X_2d = tsne.fit_transform(Xpca)

        ############################################################
        # Visualize the data
        target_names = metadata
        target_ids = range(len(target_names))

        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        for i, c, label in zip(target_ids, colors, target_names):
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
        plt.legend()
        plt.show()
       
    elif TSNE_DIM == 3:
    
        ############################################################
        # Fit and transform with a TSNE
        
        tsne = TSNE(n_components=3, random_state=0)
        # Just 2 components in order show to a 2D plot

        ############################################################
        # Project the data in 2D
        X_3d = tsne.fit_transform(Xpca)
        print(X_3d.shape)

        ############################################################
        # Visualize the data
        target_names = ['poner','remover','voltear']
        target_ids = range(len(target_names))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        for i, c, label in zip(target_ids, colors, target_names):
            ax.scatter(X_3d[y == i, 0], X_3d[y == i, 1],  X_3d[y == i, 2], c=c, label=label)
        plt.legend()
        plt.show()
    
        fig = plt.figure()
        
    else:
        print("Error: tsne dimensionality must be 2 or 3")


if __name__ == "__main__":
    main()