# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:34:36 2020

@author: cvlab
"""
from random import random
from numpy import array
from numpy import cumsum
from numpy import array_equal
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
#
#X = array([random() for _ in range(10)])
#y = cumsum(X)

#print(y[0])
#print(y[len(y)-1])

# create a random hog sequence 
def get_sequence(n_timesteps,size_elem):
    # create a sequence of random numbers in [0,1]
    size = (n_timesteps,) + size_elem
    X = np.random.random(size)
    # random classes
    y = np.random.choice([0, 1], size=n_timesteps, p=[.2, .8])
    return X, y

# create multiple samples of random hogs
def get_sequences(n_sequences, n_timesteps, size_elem):
    seqX, seqY = list(), list()
    # create and store sequences
    for _ in range(n_sequences):
        X, y = get_sequence(n_timesteps, size_elem)
        seqX.append(X)
        seqY.append(y)
    # reshape input and output for lstm
    features = size_elem[0]*size_elem[1]*size_elem[2]
    seqX = array(seqX).reshape(n_sequences, n_timesteps, features)
    seqY = array(seqY).reshape(n_sequences, n_timesteps, 1)
    return seqX, seqY


# define problem
n_timesteps = 10
n_sequences = 500
size_elem = (16,16,8) # 16x16 hog cells and 8 directions
features = size_elem[0]*size_elem[1]*size_elem[2]

# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(n_timesteps, features)))
model.add(TimeDistributed(Dense(1, activation= 'sigmoid')))
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'acc' ])
print(model.summary())

# train LSTM
X, y = get_sequences(n_sequences, 10, size_elem)
print(X.shape)
print(y.shape)

model.fit(X, y, epochs=1, batch_size=10)

# evaluate LSTM
X, y = get_sequences(100, n_timesteps, size_elem)
loss, acc = model.evaluate(X, y, verbose=0)
print( 'Loss: %f, Accuracy: %f '% (loss, acc*100))

# make predictions
for _ in range(10):
    X, y = get_sequences(1, n_timesteps, size_elem)
    
    # Deprecated removed function predict_classes
    # yhat = model.predict_classes(X, verbose=0)

    predict_x=model.predict(X) 
    yhat=np.round(predict_x,decimals=0)

    exp, pred = y.reshape(n_timesteps), yhat.reshape(n_timesteps)
    print( 'y=%s, yhat=%s, correct=%s '% (exp, pred, array_equal(exp,pred)))