# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:34:36 2020

@author: cvlab
"""
import sys
from random import random
import json

from numpy import array
from numpy import cumsum
from numpy import array_equal
import numpy as np

import sys
sys.path.append("..")

from cv_scripts.libs.mi_hog import normalize

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import keras
from keras.models import Sequential
from keras import layers
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Flatten
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, get
from tqdm import tqdm

from matplotlib import pyplot

def create_model(num_classes, input_shape, lstm_units, rec_dropout, lstm_act, lstm_rec_act, final_act, hidden_act, dropouts, hidden_dense_untis):

    inp = layers.Input(shape=input_shape)

    x = layers.Dropout(dropouts[0])(inp)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation=lstm_act,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation=lstm_act,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        #return_sequences=True,
        activation=lstm_act,
    )(x)

    x = Flatten()(x)

    x = layers.Dropout(dropouts[1])(x)
    x = layers.Dense(hidden_dense_untis, activation= hidden_act)(x)
    x = layers.Dropout(dropouts[2])(x)
    x = layers.Dense(num_classes, activation= final_act)(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)

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

    metadata_file = file1.rsplit('.',maxsplit=1)[0] + "_metadata.json"
    metadata_in = open(metadata_file,)
    metadata = json.load(metadata_in)

    class_labels = metadata['classes']

    # Normalice
    X_norm = []
    for row in range(N_SAMPLES):
        add_samples = []
        for sample in range(N_TIMESTEPS):
            ortientation_hist = X[row,sample,:,:,:]
            normalized_hist = ortientation_hist / np.max(ortientation_hist)
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
    n_sequences =  X.shape[0]
    n_timesteps =  X.shape[1]
    width = X.shape[2]
    height = X.shape[3]
    channels = X.shape[4]

    INPUT_SHAPE = (n_timesteps, width, height, channels)

    # do experiments
    NUM_EXP = 1

    lr = [0.001] * NUM_EXP
    lstm_units = [32] * NUM_EXP
    rec_drop = [0.2, 0.4, 0.5] * 2
    lstm_act = ['relu'] * NUM_EXP
    lstm_rec_act = ['hard_sigmoid'] * NUM_EXP
    final_act = ['softmax'] * NUM_EXP
    hidden_act = ['sigmoid'] * NUM_EXP
    dropouts = [[0.5,0.3,0.2]] * NUM_EXP
    hidden_dense_untis = [32] * NUM_EXP

    optimizers = ['adam'] * NUM_EXP
    losses = ['categorical_crossentropy'] * NUM_EXP
    epochs = [50] * NUM_EXP

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
        history = model.fit(trainX, trainy, validation_data=(valX, valy), epochs=epochs[i], batch_size=BATCH_SIZE, 
                                callbacks=[es, reduce_lr], shuffle=True, verbose=1)


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

        to_append = {'model': model, 'history': metrics, 'vis': to_vis}

        models_metrics.append(to_append)

        t.update()

    # dumps results
    out_file = open("out_model_metrics.json", "w")
    json.dump(str(models_metrics), out_file, indent=1)

    model.save('out_model.h5')

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

    
    pyplot.show()

if __name__ == "__main__":
    main()