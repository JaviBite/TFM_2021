'''
Fully connected neural network for class prediction
'''

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np

num_classes = 4 # Number of different actions (plus an optional 'others' one)
num_tiles = 16*16 # Number of inputs

def load_hog():
    print('Loading dataset...')
    return

# load HOG
(x_train, y_train), (x_test, y_test) = load_hog()
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

# random permutation of training data
np.random.seed(0)
p = np.arange(x_train.shape[0])
np.random.shuffle(p)
x_train = x_train[p]
y_train = y_train[p]

# Stop training when validation error no longer improves
earlystop=EarlyStopping(monitor='val_loss', patience=5, 
                        verbose=1, mode='auto')

# Model definition
model = Sequential()

# Single layer perceptron
model.add(Dense(num_classes, activation='sigmoid', input_shape=(num_tiles,)))
