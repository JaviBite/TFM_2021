# -*- coding: utf-8 -*-
"""
Utility functions for convolutional neural network training

@author: Miguel
"""

import json
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import itertools
import keras.utils
from keras.models import Sequential
from keras.metrics import top_k_categorical_accuracy
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

def load_data(file,window):
    
    """
    Returns normalized and separated training and test values and tags.
    Also returns the number of classes, the classes themselves, and the number
    of windows. If window is false, only 1 window is returned and, if the data
    has more than one window, they are considered different samples. Otherwise,
    each sample retains its number of windows, so they can be treated accordingly.
    """
    
    metadata_path = file+'_metadata.json'
    train_path = file+'_train.npz'
    test_path = file+'_test.npz'

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
    windows = train_data['a'].shape[1]
    dim = train_data['a'].shape[2]
    dirs = train_data['a'].shape[4]
    
    Ytrain = train_data['b'][:]
    
    if window:
        Xtrain = np.reshape(train_data['a'],(train_samples,windows,dim,dim,dirs))
    else :
        Xtrain = np.reshape(train_data['a'],(train_samples*windows,dim,dim,dirs))
        Ytrain = np.array([val for val in Ytrain for _ in range(windows)])
    # end if
    
    print('Loading test data...')
    test_data = np.load(test_path)
    
    test_samples = test_data['a'].shape[0]
    
    Ytest = test_data['b'][:]
    
    if window:
        Xtest = np.reshape(test_data['a'],(test_samples,windows,dim,dim,dirs))
    else:
        Xtest = np.reshape(test_data['a'],(test_samples*windows,dim,dim,dirs))
        Ytest = np.array([val for val in Ytest for _ in range(windows)])
    # end if
    
    AllX = np.append(Xtrain,Xtest,axis=0)
    #AllX = AllX/np.linalg.norm(AllX)
    #AllX = AllX/np.max(AllX)
    #AllX = (AllX-np.min(AllX))/(np.max(AllX)-np.min(AllX))
    AllX = (AllX-np.mean(AllX))/np.std(AllX)
    
    Xtrain = AllX[:train_samples*windows]
    Xtest = AllX[train_samples*windows:]
    
    if window:
        # No matter if we want to test with window data or not, training will
        # always consider each window a different sample
        Xtrain = np.reshape(Xtrain,(train_samples*windows,dim,dim,dirs))

    return Xtrain,Xtest,Ytrain,Ytest,num_classes,classes,windows

def kfold(X,Y,num_classes,k_folds):
    
    """
    Separates the training and validation data following the k-fold method
    """
            
    # print('Performing k-fold separation of data')
    
    k_folds_x_train = [[]]*k_folds
    k_folds_x_val = [[]]*k_folds
    k_folds_y_train = [[]]*k_folds
    k_folds_y_val = [[]]*k_folds
    
    dim = X.shape[1]
    dirs = X.shape[3]
    
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

def getTop2Acc(Y_true,Y_pred):
    return np.round(np.mean(top_k_categorical_accuracy(Y_true,Y_pred,k = 2)),3)

def plot_confusion_matrix(y_test, y_pred, num_classes):
    
    classes = range(num_classes)
    y_tst = [np.argmax(y) for y in y_test]
    y_prd = [np.argmax(y) for y in y_pred]

    cm = confusion_matrix(y_tst, y_prd)
    cm = np.round(cm/np.sum(cm,axis=0),3)

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
                  metrics=['accuracy'])
    return model
# end def

def trainloop(Xtrain, Ytrain, num_classes, score_threshold, input_shape, 
              model_path):
    
    # ---------------------------------
    # Parameter combinators
    
    # Configure the vars and lists below 
    # and the program will combine them 
    # all to find the best combination!
    
    # (Yeah brute force isn't very fancy
    # or fast, but you gotta do what you
    # gotta do :/ )
    # ---------------------------------
    
    # Number of iterations for averaging
    # ( the 'k' value for  k-fold cross-validation )
    num_it = 4
    
    # Earlystopping patience (epochs without improvement)
    patiences = [2,10]
    
    # Dimensions of kernel (only square kernels will be tested)
    # Beware bigger k-sizes depending on the number of conv. layers
    k_sizes = [3]
    
    # Activation function for the convolutional layers
    activations = ['tanh','relu']
    
    # Number of convolutional layers. Second layer onward will always have a
    # 3 wide, 3 tall kernel.
    num_conv = [1,2,3]
    
    # Add a pooling layer or not. If added, it will be (2,2) 
    # (halves the input size in every dimension)
    pooling = [True,False]
    
    # Add a hidden linear layer or not after the convolution and pooling.
    hidden = [True,False]
    
    # Dropout levels
    dropouts = [0.1,0.3]
    
    kf_x_train, kf_x_val, kf_y_train, kf_y_val = kfold(Xtrain,Ytrain,
                                                       num_classes,
                                                       num_it)
    
    Ytrain = keras.utils.to_categorical(Ytrain,num_classes)
    
    num_model = 1
    
    out_json  = {}
    models = []
    
    for pat in patiences:
        
         # Stop training when validation error no longer decreases
        earlystop=EarlyStopping(monitor='val_loss', patience=pat, 
                                verbose=0, mode='auto')
        
        for size in k_sizes:
            for act in activations:
                for conv in num_conv:
                    for pool in pooling:
                        for hid in hidden:
                            for drop in dropouts:
                                
                                config = ((size,size),act,conv,pat,drop,pool,hid)
                                total_score = 0
                
                                for i in range(num_it):
                                    
                                    print('Iteration: '+str(i)+'...')
                                    
                                    x_train = kf_x_train[i]
                                    x_val = kf_x_val[i]
                                    y_train = kf_y_train[i]
                                    y_val = kf_y_val[i]
                                
                                    # convert class vectors to binary class matrices
                                    y_train = keras.utils.to_categorical(y_train,num_classes)
                                    y_val  = keras.utils.to_categorical(y_val,num_classes)
                                    
                                    model = getUntrained2DConv(size,act,
                                                               conv,drop,pool,hid,
                                                               input_shape,num_classes)
                                
                                    # Model training
                                    model.fit(x_train, y_train,
                                              batch_size=8,
                                              epochs=50,
                                              validation_split=0.1,
                                              callbacks=[earlystop],
                                              verbose=False)
                            
                                    # Model evaluation
                                    val_score = model.evaluate(x_val, y_val, verbose=0)
                                    print('%s %2.2f%s' % ('Accuracy test:  ', 100*val_score[1], '%'))
                                    total_score = total_score + val_score[1]
                                    
                                # end for  i
                                
                                av_score = total_score / num_it
                                print('Finished testing model: '+str(config))
                                print('Average accuracy: '+str(100*av_score)+'%')
                                
                                if av_score > score_threshold:
                                    
                                    out_json[num_model] = {'Kernel': (size,size),
                                                           'Act. f.': act,
                                                           'Num. Conv.': conv,
                                                           'Patience': pat,
                                                           'Dropout': drop,
                                                           'Pooling': pool,
                                                           'Hidden': hid,
                                                           'Avg. acc.': av_score,
                                                           }
                                    
                                    print('Model accepted, retraining.')
                                    
                                    model = getUntrained2DConv(size,act,
                                                               conv,drop,pool,hid,
                                                               input_shape,num_classes)
                                    
                                    model.fit(Xtrain,Ytrain,
                                            batch_size=4,
                                            epochs=100,
                                            validation_split=0.15,
                                            callbacks=[earlystop],
                                            verbose=True)
                                    
                                    num_model = num_model+1
                                    models.append(model)
                                    
                                    print('Model approved')
                                    
                                else:
                                    print('Model discarded')
                    
                            # end for drop
                        #end for hid
                    #end for pool
                # end for dcv
            #end for act
        # end for size
    # end for pat
    
    return out_json, models