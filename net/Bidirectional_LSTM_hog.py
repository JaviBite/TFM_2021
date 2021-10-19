# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:34:36 2020

@author: cvlab
"""
import sys
from random import random

from numpy import array
from numpy import cumsum
from numpy import array_equal
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Flatten
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping

from matplotlib import pyplot


# Load data
file1 = sys.argv[1]
files = np.load(file1, allow_pickle=True)
X, labels = files['a'], files['b']

num_classes = 2
y = []
for yi in labels:
    to_append = np.zeros(num_classes)
    to_append[yi] = 1
    y.append(to_append)

y = np.array(y).reshape((len(labels), num_classes))

print("X Shape: ", X.shape)
print("Y Shape: ", y.shape)

val_percent = 0.2
train_percent = 1 - val_percent

n_train = int(len(labels) * train_percent)

trainX, valX = X[:n_train, :, :], X[n_train:, :, :]
trainy, valy = y[:n_train, :], y[n_train:, :]

# define problem
n_timesteps =  X.shape[1]
n_sequences =  X.shape[0]
features = X.shape[2]

print(features)
# define LSTM
model = Sequential()
model.add(Dropout(0.5, input_shape=(n_timesteps, features)))
model.add(Bidirectional(LSTM(50, return_sequences=False, activation='relu')))
model.add(Dropout(0.5))
#model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation = "relu"))
model.compile(loss= 'mse' , optimizer= 'adam' , metrics=[ 'acc' ])
print(model.summary())

# train LSTM
#X, y = get_sequences(n_sequences, 10, size_elem)

print(X.shape)
print(y.shape)

print("Num sequences: ",len(labels))
for i in range(2):
    count = np.sum(labels == i)
    print("Class ",i,":", count)

# Early Estopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = model.fit(trainX, trainy, validation_data=(valX, valy), epochs=5, batch_size=10, callbacks=[es])

model.save('out_model.h5')

# evaluate LSTM
#X, y = get_sequences(100, n_timesteps, size_elem)
loss, acc = model.evaluate(valX, valy, verbose=0)
print( 'Loss: %f, Accuracy: %f '% (loss, acc*100))

# make predictions
#X, y = get_sequences(1, n_timesteps, size_elem)

# Deprecated removed function predict_classes
# yhat = model.predict_classes(X, verbose=0)

predict_x=model.predict(X[0:10,:,:]) 
yhat=np.round(predict_x,decimals=0)

#exp, pred = y.reshape(n_timesteps), yhat.reshape(n_timesteps)

exp, pred = y[0:10], yhat
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