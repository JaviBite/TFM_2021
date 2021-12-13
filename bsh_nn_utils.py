# -*- coding: utf-8 -*-
"""
Utility functions for fully-connected neural network training

@author: Miguel
"""

import json
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import itertools
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.optimizers import Adam, Adadelta

def load_data(path,file,num_tiles):
    
    train_path = path+file+'_train.npz'
    test_path = path+file+'_test.npz'
    metadata_path = path+file+'_metadata.json'

    print('Opening metadata file...')
    mdf = open(metadata_path,)
    metadata = json.load(mdf)
    classes = metadata['classes']
    num_classes = len(classes)
    
    print('Read ',num_classes,' classes')
    print(classes)
    print('Loading training data...')
    train_data = np.load(train_path)
    
    Xtrain = np.reshape(train_data['a'],(train_data['a'].shape[0],num_tiles))
    Ytrain = train_data['b'][:]
    
    print('Loading test data...')
    test_data = np.load(test_path)
    
    Xtest = np.reshape(test_data['a'],(test_data['a'].shape[0],num_tiles))
    Ytest = test_data['b'][:]
    
    AllX = np.append(Xtrain,Xtest,axis=0)
    #AllX = AllX/np.linalg.norm(AllX)
    #AllX = AllX/np.max(AllX)
    #AllX = (AllX-np.min(AllX))/(np.max(AllX)-np.min(AllX))
    AllX = (AllX-np.mean(AllX))/np.std(AllX)

    return Xtrain,Xtest,Ytrain,Ytest,num_classes,classes

def load_3d_data(path,file,dim,dirs):
    
    metadata_path = path+file+'_metadata.json'
    train_path = path+file+'_train.npz'
    test_path = path+file+'_test.npz'

    print('Opening metadata file...')
    mdf = open(metadata_path,)
    metadata = json.load(mdf)
    classes = metadata['classes']
    num_classes = len(classes)
    
    print('Read ',num_classes,' classes')
    print(classes)
    print('Loading training data...')
    train_data = np.load(train_path)
    
    train_samples = train_data['a'].shape[0]
    Xtrain = np.reshape(train_data['a'],(train_samples,dim,dim,dirs))
    Ytrain = train_data['b'][:]
    
    print('Loading test data...')
    test_data = np.load(test_path)
    
    test_samples = test_data['a'].shape[0]
    Xtest = np.reshape(test_data['a'],(test_samples,dim,dim,dirs))
    Ytest = test_data['b'][:]
    
    #AllX = np.append(Xtrain,Xtest,axis=0)
    AllX = Xtrain
    #AllX = AllX/np.linalg.norm(AllX)
    #AllX = AllX/np.max(AllX)
    #AllX = (AllX-np.min(AllX))/(np.max(AllX)-np.min(AllX))
    AllX = (AllX-np.mean(AllX))/np.std(AllX)
    
    # Xtrain = AllX[:train_samples]
    # Xtest = AllX[train_samples:]
    Xtrain = AllX

    return Xtrain,Xtest,Ytrain,Ytest,num_classes,classes

def load_3d_w_data(path,file,windows,dim,dirs):
    
    metadata_path = path+file+'_metadata.json'
    train_path = path+file+'_train.npz'
    test_path = path+file+'_test.npz'

    print('Opening metadata file...')
    mdf = open(metadata_path,)
    metadata = json.load(mdf)
    classes = metadata['classes']
    num_classes = len(classes)
    
    print('Read ',num_classes,' classes')
    print(classes)
    print('Loading training data...')
    train_data = np.load(train_path)
    
    Xtrain = train_data['a']
    train_samples = Xtrain.shape[0]
    Ytrain = train_data['b']
    
    print('Loading test data...')
    test_data = np.load(test_path)
    
    Xtest = test_data['a']
    test_samples = Xtest.shape[0]
    Ytest = test_data['b']
    
    AllX = np.append(Xtrain,Xtest,axis=0)
    #AllX = AllX/np.linalg.norm(AllX)
    #AllX = AllX/np.max(AllX)
    #AllX = (AllX-np.min(AllX))/(np.max(AllX)-np.min(AllX))
    AllX = (AllX-np.mean(AllX))/np.std(AllX)
    
    Xtrain = AllX[:train_samples]
    Xtest = AllX[train_samples:]

    return Xtrain,Xtest,Ytrain,Ytest,num_classes,classes

def load_3d_flat_w_data(path,file,dim,dirs):
    
    metadata_path = path+file+'_metadata.json'
    train_path = path+file+'_train.npz'
    test_path = path+file+'_test.npz'

    print('Opening metadata file...')
    mdf = open(metadata_path,)
    metadata = json.load(mdf)
    classes = metadata['classes']
    num_classes = len(classes)
    
    print('Read ',num_classes,' classes')
    print(classes)
    print('Loading training data...')
    train_data = np.load(train_path)
    
    train_samples = train_data['a'].shape[0]
    print(train_data['a'].shape)
    windows = train_data['a'].shape[1]
    Xtrain = np.reshape(train_data['a'],(train_samples*windows,dim,dim,dirs))
    Ytrain = train_data['b'][:]
    Ytrain = np.array([val for val in Ytrain for _ in range(windows)])
    
    print('Loading test data...')
    test_data = np.load(test_path)
    
    test_samples = test_data['a'].shape[0]
    print(test_data['a'].shape)
    Xtest = np.reshape(test_data['a'],(test_samples*windows,dim,dim,dirs))
    Ytest = test_data['b'][:]
    Ytest = np.array([val for val in Ytest for _ in range(windows)])
    
    AllX = np.append(Xtrain,Xtest,axis=0)
    #AllX = AllX/np.linalg.norm(AllX)
    #AllX = AllX/np.max(AllX)
    #AllX = (AllX-np.min(AllX))/(np.max(AllX)-np.min(AllX))
    AllX = (AllX-np.mean(AllX))/np.std(AllX)
    
    Xtrain = AllX[:train_samples*windows]
    Xtest = AllX[train_samples*windows:]

    return Xtrain,Xtest,Ytrain,Ytest,num_classes,classes

# Separates training and test data and tags. Each class is separately
# shuffled and partitioned to ensure balance between classes and add
# a randomizing factor to the learning, in order to help find the best
# network among the possible training results.
def separate_val_data(X,Y,num_classes,num_tiles):
    
    x_train = x_val = np.array([[0]*num_tiles])
    y_train = y_val = np.array([])
    
    # seed for randomizer
    np.random.seed(0)
    
    print(X.shape)
    
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
        val_samples = num_samples//10+1
        train_samples = num_samples-val_samples
        # print('Class: ',i)
        # print('Train samples: ',train_samples,', Test samples: ', test_samples)
        x_val = np.append(x_val,x[:val_samples],axis=0)
        y_val = np.append(y_val,y[:val_samples])
        x_train = np.append(x_train,x[val_samples:],axis=0)
        y_train = np.append(y_train,y[val_samples:])
    
    x_val = x_val[1:]
    x_train = x_train[1:]
        
    # random permutation for training data
    p = np.arange(x_train.shape[0])
    np.random.shuffle(p)
    x_train = x_train[p]
    y_train = y_train[p]
    
    # print('Done')
    
    return (x_train,y_train),(x_val,y_val)

# Separates the training and validation data following the k-fold method.

def kfold(X,Y,num_classes,dim,dirs,k_folds):
            
    # print('Performing k-fold separation of data')
    
    k_folds_x_train = [[]]*k_folds
    k_folds_x_val = [[]]*k_folds
    k_folds_y_train = [[]]*k_folds
    k_folds_y_val = [[]]*k_folds
    
    for k in range(k_folds):
        
        x_train_k = x_val_k = np.zeros((1,dim,dim,dirs))
        y_train_k = y_val_k = []
    
        for i in range(num_classes):
            
            x = X[Y==i]
            y = Y[Y==i]
            
            num_samples = len(y)
            k_samples = num_samples//k_folds
            
            if k == 0:
                x_val = x[:k_samples]
                y_val = y[:k_samples]
                x_train = x[k_samples:]
                y_train = y[k_samples:]
            elif k == k_folds-1:
                x_val = x[num_samples-k_samples:]
                y_val = y[num_samples-k_samples:]
                x_train = x[:num_samples-k_samples]
                y_train = y[:num_samples-k_samples]
            else:
                x_val = x[k_samples*k:k_samples*(k+1)]
                y_val = y[k_samples*k:k_samples*(k+1)]
                x_train = np.append(x[:k_samples*k],x[k_samples*(k+1):],axis=0)
                y_train = np.append(y[:k_samples*k],y[k_samples*(k+1):],axis=0)
            
            x_train_k = np.append(x_train_k,x_train,axis=0) 
            y_train_k = np.append(y_train_k,y_train,axis=0) 
            x_val_k = np.append(x_val_k,x_val,axis=0) 
            y_val_k = np.append(y_val_k,y_val,axis=0) 
        
        k_folds_x_train[k] = x_train_k[1:]
        k_folds_x_val[k] = x_val_k[1:]
        k_folds_y_train[k] = y_train_k
        k_folds_y_val[k] = y_val_k
            
    # print('Done')
    
    return k_folds_x_train, k_folds_x_val, k_folds_y_train, k_folds_y_val

def getF1scores(y_test, y_pred):
    
    y_tst = [np.argmax(y) for y in y_test]
    y_prd = [np.argmax(y) for y in y_pred]
    
    f1s = np.round(f1_score(y_tst,y_prd,average=None),3)
    
    return f1s

def getAccuracy(y_test, y_pred):
    
    y_tst = [np.argmax(y) for y in y_test]
    y_prd = [np.argmax(y) for y in y_pred]
    
    acc = np.round(accuracy_score(y_tst,y_prd),3)
    
    return acc

def plot_confusion_matrix(y_test, y_pred, num_classes):
    
    classes = range(num_classes)
    y_tst = [np.argmax(y) for y in y_test]
    y_prd = [np.argmax(y) for y in y_pred]

    cm = confusion_matrix(y_tst, y_prd)
    #cm = np.round(cm/np.sum(cm,axis=1),3)

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

def getUntrained2DConv(size,act,conv,drop,pool,hidden,input_shape,num_classes):
    model = Sequential()
    model.add(Input(input_shape))
    #model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(size,size),
                     activation=act,data_format='channels_last'))
    if drop > 0:
        model.add(Dropout(drop))
    if conv>1:
        model.add(Conv2D(32, kernel_size=(3,3),
                         activation=act,data_format='channels_last'))
        if drop > 0:
            model.add(Dropout(drop))
        if conv>2:
            model.add(Conv2D(32, kernel_size=(3,3),
                             activation=act,data_format='channels_last'))
            if drop > 0:
                model.add(Dropout(drop))
    if pool:
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    if hidden:
        model.add(Dense(128, activation='relu'))
        if drop > 0:
            model.add(Dropout(min(drop*2,0.5)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy','categorical_accuracy'])
    return model
# end def