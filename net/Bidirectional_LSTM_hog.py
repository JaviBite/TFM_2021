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
    y = np.random.choice([0, 1], size=1, p=[.2, .8])
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
    seqY = array(seqY).reshape(n_sequences, 1,)
    return seqX, seqY


# Load data
file1 = '../out.npz'
files = np.load(file1, allow_pickle=True)
X, y = files['a'], files['b']
y = y.reshape((len(y), 1,))

# define problem
n_timesteps =  X.shape[1]
n_sequences =  X.shape[0]
size_elem = (16,16,3,3,9) # 16x16 hog cells and 8 directions
features = X.shape[2]

print(features)
# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=False), input_shape=(n_timesteps, features)))
#model.add(TimeDistributed(Dense(1, activation= 'sigmoid')))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'acc' ])
print(model.summary())

# train LSTM
#X, y = get_sequences(n_sequences, 10, size_elem)

print(X.shape)
print(y.shape)

print("Num sequences: ",len(y))
for i in range(2):
    count = np.sum(y == i)
    print("Class ",i,":", count)

model.fit(X, y, epochs=10, batch_size=10)

# evaluate LSTM
#X, y = get_sequences(100, n_timesteps, size_elem)
loss, acc = model.evaluate(X, y, verbose=0)
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