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

from keras.models import Sequential
from keras import regularizers
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Flatten
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, get
from tqdm import tqdm

from matplotlib import pyplot

def create_model(num_classes, input_shape, lstm_units, rec_dropout, lstm_act, lstm_rec_act, final_act, hidden_act, dropouts, hidden_dense_untis):

    model = Sequential()
    model.add(Dropout(dropouts[0], input_shape=input_shape)) # (n_timesteps, n_features)
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=False, activation=lstm_act, recurrent_activation=lstm_rec_act, recurrent_dropout=rec_dropout, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))))
    model.add(Dropout(dropouts[1]))
    #model.add(Flatten())
    model.add(Dense(hidden_dense_untis, activation = hidden_act, kernel_regularizer=regularizers.l1_l2()))
    model.add(Dropout(dropouts[2]))
    model.add(Dense(num_classes, activation = final_act, kernel_regularizer=regularizers.l2()))

    return model

def main():
    # Load data
    file1 = sys.argv[1]
    files = np.load(file1, allow_pickle=True)
    X, labels = files['a'], files['b']

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

    # do experiments
    NUM_EXP = 6

    lr = [0.001] * NUM_EXP
    lstm_units = [32] * NUM_EXP
    rec_drop = [0.2, 0.4, 0.5] * 2
    lstm_act = ['tanh'] * NUM_EXP
    lstm_rec_act = ['hard_sigmoid'] * NUM_EXP
    final_act = ['softmax'] * NUM_EXP
    hidden_act = ['sigmoid'] * NUM_EXP
    dropouts = [[0.5,0.5,0.2],[0.5,0.3,0.2],[0,0.5,0.3]] * 2
    hidden_dense_untis = [32] * NUM_EXP

    optimizers = ['adam'] * NUM_EXP
    losses = ['categorical_crossentropy'] * NUM_EXP
    epochs = [30] * NUM_EXP

    to_vis = ['rec_drop','dropouts']

    BATCH_SIZE = 10

    models_metrics = []

    # Run the combinations
    t = tqdm(total=NUM_EXP)
    for i in range(NUM_EXP):

        model = create_model(N_CLASSES, INPUT_SHAPE, lstm_units[i], rec_drop[i], lstm_act[i], 
                                lstm_rec_act[i], final_act[i], hidden_act[i], dropouts[i], hidden_dense_untis[i])
        opt = get(optimizers[i])
        opt.learning_rate = lr[i]
        model.compile(loss= losses[i] , optimizer= opt , metrics=[ 'acc' ])
        #print(model.summary())

        # Early Estopping
        es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5)

        start = time.time()
        history = model.fit(trainX, trainy, validation_data=(valX, valy), epochs=epochs[i], batch_size=BATCH_SIZE, 
                                callbacks=[es, reduce_lr], shuffle=True, verbose=0)

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_time = {'hours': hours, 'minutes': minutes, 'seconds': seconds}

        model = {'lr': lr[i],
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

        to_append = {'model': model, 'history': metrics, 'vis': to_vis, 'etime': elapsed_time}

        models_metrics.append(to_append)

        t.update()

    # dumps results
    out_file = open("out_bimodel_metrics.json", "w")
    json.dump(models_metrics, out_file, indent=1)

    
    if False:

        model.save('out_model.h5')

        # evaluate LSTM
        #X, y = get_sequences(100, n_timesteps, size_elem)
        loss, acc = model.evaluate(valX, valy, verbose=0)
        print( 'Loss: %f, Accuracy: %f '% (loss, acc*100))

        # make predictions
        #X, y = get_sequences(1, n_timesteps, size_elem)

        # Deprecated removed function predict_classes
        # yhat = model.predict_classes(X, verbose=0)

        predict_x=model.predict(valX) 
        yhat=np.round(predict_x,decimals=0)

        #exp, pred = y.reshape(n_timesteps), yhat.reshape(n_timesteps)

        exp, pred = valy[0:10], yhat
        print( 'y=%s, yhat=%s, correct=%s '% (exp, pred, array_equal(exp,pred)))

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

        pyplot.show()

if __name__ == "__main__":
    main()