# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:34:36 2020

@author: cvlab
"""
import sys
from random import random
import json, time

from numpy import array
from numpy import cumsum
from numpy import array_equal
import numpy as np

import sys
sys.path.append("..")

from cv_scripts.libs.mi_hog import normalize

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from keras.models import Sequential
from keras.losses import categorical_crossentropy as cc
from keras.regularizers import l1, l2, l1_l2
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Flatten
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, get
from tqdm import tqdm
from sklearn.model_selection import KFold

from matplotlib import pyplot

def create_model(num_classes, input_shape, lstm_units, rec_dropout, lstm_act, lstm_rec_act, final_act, hidden_act, dropouts, hidden_dense_untis):

    model = Sequential()
    model.add(Dropout(dropouts[0], input_shape=input_shape)) # (n_timesteps, n_features)
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=False, activation=lstm_act, recurrent_activation=lstm_rec_act, recurrent_dropout=rec_dropout, \
                             kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))))
    model.add(Dropout(dropouts[1]))
    model.add(Flatten())
    model.add(Dense(hidden_dense_untis, activation = hidden_act, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(dropouts[2]))
    model.add(Dense(num_classes, activation = final_act))

    return model

def main():
    # Load data
    file1 = sys.argv[1]
    files = np.load(file1, allow_pickle=True)
    X, labels = files['a'], files['b']

    if '_train' in file1:
        metadata_file = file1.rsplit('.',maxsplit=1)[0][:-6] + "_metadata.json"
    else:
        metadata_file = file1.rsplit('.',maxsplit=1)[0] + "_metadata.json"
    metadata_in = open(metadata_file,)
    metadata = json.load(metadata_in)

    class_labels = metadata['classes']

    N_CLASSES = np.max(labels) + 1
    N_SAMPLES = X.shape[0]
    N_TIMESTEPS = X.shape[1]

    HOG_H = X.shape[2]
    HOG_W = X.shape[3]
    ORIENTATIONS = X.shape[4]

    # Normalice
    X_norm = []
    for row in range(N_SAMPLES):
        add_samples = []
        for sample in range(N_TIMESTEPS):
            ortientation_hist = X[row,sample,:,:,:]
            normalized_hist = normalize(ortientation_hist)
            add_samples.append(normalized_hist)
        X_norm.append(np.array(add_samples))

    X_norm = np.array(X_norm)
    X = X_norm
    
    # Ravel features into an array
    #X = X.reshape(N_SAMPLES,N_TIMESTEPS,HOG_H*HOG_W*ORIENTATIONS)
    y = []
    
    count_classes = [0] * N_CLASSES
    for yi in labels:
        to_append = np.zeros(N_CLASSES)
        to_append[yi] = 1
        count_classes[yi] += 1
        y.append(to_append)
        
    print("Count classes:",count_classes)

    y = np.array(y).reshape((len(labels), N_CLASSES))

    print("X Shape: ", X.shape)
    print("Y Shape: ", y.shape)

    val_percent = 0.2
    trainX, valX, trainy, valy = train_test_split(X, y, test_size=val_percent, stratify=y)

    # define problem
    n_timesteps =  X.shape[1]
    n_sequences =  X.shape[0]
    n_features = X.shape[2]

    INPUT_SHAPE = (n_timesteps, n_features)

    lr = [0.001]
    lstm_units = [32]
    rec_drop = [0.3]
    lstm_act = ['tanh']
    lstm_rec_act = ['hard_sigmoid']
    final_act = ['softmax']
    hidden_act = ['sigmoid']
    dropouts = [[0.3,0.5,0.3]]
    hidden_dense_untis = [64]

    optimizers = ['adam']
    losses = ['categorical_crossentropy']
    epochs = [30]

    BATCH_SIZE = 10
    i = 0

    model = create_model(N_CLASSES, INPUT_SHAPE, lstm_units[i], rec_drop[i], lstm_act[i], 
                            lstm_rec_act[i], final_act[i], hidden_act[i], dropouts[i], hidden_dense_untis[i])
    opt = get(optimizers[i])
    opt.learning_rate = lr[i]
    model.compile(loss= losses[i] , optimizer= opt , metrics=[ 'acc' ])
    #print(model.summary())

    # Early Estopping
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=5, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, val in kfold.split(X, y):

        start = time.time()
        history = model.fit(X[train], y[train], validation_data=(X[val], y[val]), epochs=epochs[i], batch_size=BATCH_SIZE, 
                                callbacks=[es, reduce_lr], shuffle=True, verbose=1)
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_time = {'hours': hours, 'minutes': minutes, 'seconds': seconds}

        fold_no = fold_no + 1

        print("Elapsed time: ", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

        model_json = {'lr': lr[i],
                    'lstm_units': lstm_units[i],
                    'rec_drop': rec_drop[i],
                    'lstm_act': lstm_act[i],
                    'lstm_rec_act': lstm_rec_act[i],
                    'final_act': final_act[i],
                    'hidden_act': hidden_act[i],
                    'dropouts': dropouts[i],
                    'hidden_dense_untis': hidden_dense_untis[i],
                    'optimizers': optimizers[i],
                    'losses': losses[i],
                    'epochs': epochs[i]
        }

        metrics = history.history

        lr_list = []
        for f in metrics['lr']:
            lr_list.append(float(f))
        metrics['lr'] = lr_list
        models_metrics = [{'model': model_json, 'history': metrics, 'etime': elapsed_time}]

    # dumps results
    out_file = open("out_model_metrics.json", "w")
    json.dump(models_metrics, out_file, indent=1)

    model.save('out_model_bilstm.h5')

    # evaluate LSTM
    #X, y = get_sequences(100, n_timesteps, size_elem)
    loss, acc = model.evaluate(valX, valy, verbose=0)
    print( 'Loss: %f, Accuracy: %f '% (loss, acc*100))

    predict_x=model.predict(valX) 
    yhat=np.round(predict_x,decimals=0)

    fig, axs = pyplot.subplots(2, 1, constrained_layout=True)
    axs[0].set_title('Loss')
    axs[0].plot(history.history['loss'], label='train')
    axs[0].plot(history.history['val_loss'], label='test')
    axs[0].legend()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')
    axs[1].plot(history.history['acc'], label='train')
    axs[1].plot(history.history['val_acc'], label='test')
    axs[1].legend()
    axs[1].set_title('Accuracy')
    axs[1].set_ylabel('Accuracy')


    # Confusion matrix
    matrix = confusion_matrix(valy.argmax(axis=1), predict_x.argmax(axis=1), normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_labels)
    disp.plot()

    # Loss histogram
    losses = []
    for i in range(len(valX)):
        losses.append(cc(valy[i], yhat[i]))

    fig, axs = pyplot.subplots(1, 1, constrained_layout=True)
    axs.hist(losses)
    axs.set_title('Loss histogram')
    axs.set_xlabel('Loss value')
    axs.set_ylabel('Frecuency')
    
    pyplot.show()

if __name__ == "__main__":
    main()