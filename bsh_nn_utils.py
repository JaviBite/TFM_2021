# -*- coding: utf-8 -*-
"""
Utility functions for fully-connected neural network training

@author: Miguel
"""

import json
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# Returns a normalized 2D array for the training data X , and 
# a 1D array for the class tags, as well as the number of classes red
# from the metadata file.
def load_data(path,file,num_tiles):
    
    data_path = path+file+'.npz'
    metadata_path = path+file+'_metadata.json'

    print('Opening metadata file...')
    mdf = open(metadata_path,)
    metadata = json.load(mdf)
    classes = metadata['classes']
    num_classes = len(classes)
    
    print('Read ',num_classes,' classes')
    print(classes)
    print('Loading training data...')
    data = np.load(data_path)
    
    X = np.reshape(data['a'],(data['a'].shape[0],num_tiles))
    X = (X - X.min())/(X.max()-X.min()) #Normalize the data
    Y = data['b'][:]

    return X,Y,num_classes

# Separates training and test data and tags. Each class is separately
# shuffled and partitioned to ensure balance between classes and add
# a randomizing factor to the learning, in order to help find the best
# network among the possible training results.
def separate_test_data(X,Y,num_classes,num_tiles):
    
    x_train = x_test = np.array([[0]*num_tiles])
    y_train = y_test = np.array([])
    
    # seed for randomizer
    np.random.seed(0)
    
    # print('Separating test data...')
    for i in range(num_classes):
        
        x = X[Y==i]
        y = Y[Y==i]
        
        # random permutation for test data separation
        p = np.arange(x.shape[0])
        np.random.shuffle(p)
        x = x[p]
        y = y[p]
        
        num_samples = len(y)
        test_samples = num_samples//7+1
        train_samples = num_samples-test_samples
        # print('Class: ',i)
        # print('Train samples: ',train_samples,', Test samples: ', test_samples)
        x_test = np.append(x_test,x[:test_samples],axis=0)
        y_test = np.append(y_test,y[:test_samples])
        x_train = np.append(x_train,x[test_samples:],axis=0)
        y_train = np.append(y_train,y[test_samples:])
    
    x_test = x_test[1:]
    x_train = x_train[1:]
        
    # random permutation for training data
    p = np.arange(x_train.shape[0])
    np.random.shuffle(p)
    x_train = x_train[p]
    y_train = y_train[p]
    
    # print('Done')
    
    return (x_train,y_train),(x_test,y_test)

def plot_confusion_matrix(y_test, y_pred, num_classes):
    
    classes = range(num_classes)
    y_tst = [np.argmax(y) for y in y_test]
    y_prd = [np.argmax(y) for y in y_pred]

    cm = confusion_matrix(y_tst, y_prd)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.show()
    
# Placeholder
def perform_tSNE():
    
    print('Performing PCA & tSNE')
    # pca = decomposition.PCA(n_components=50)
    # pca.fit(x_train)
    # x_train_pca = pca.transform(x_train)
    # tsne = TSNE(n_components=3, random_state=0)
    # x_train_3d = tsne.fit_transform(x_train_pca)

    # pca = decomposition.PCA(n_components=50)
    # pca.fit(x_test)
    # x_test_pca = pca.transform(x_test)
    # tsne = TSNE(n_components=3, random_state=0)
    # x_test_3d = tsne.fit_transform(x_test_pca)
    # print(x_train_3d.shape)
    # print(x_test_3d.shape)

    # input_shape = (3,)
    
    return