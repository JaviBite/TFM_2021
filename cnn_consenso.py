# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 09:44:33 2021

@author: Miguel
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta

import bsh_nn_utils as util
import numpy as np

# ---------------------------------
# Data metrics for input and output
# ---------------------------------

img_dim = 15
hog_dirs = 9
input_shape = (img_dim,img_dim,hog_dirs,1) # Number of inputs for the neural network

path = 'D:\\GI Lab\\Pruebas_acc_v2\\'
file = 'out_flow_f8_mf500'

Xtrain, Xtest, Ytrain, Ytest, num_classes, classes = util.load_3d_data(path,file,img_dim,hog_dirs)

Ytrain = keras.utils.to_categorical(Ytrain, num_classes)
Ytest = keras.utils.to_categorical(Ytest, num_classes)

model1 = keras.models.load_model(path+'model_1'+file[8:]+'.h5')
model2 = keras.models.load_model(path+'model_2'+file[8:]+'.h5')
model3 = keras.models.load_model(path+'model_3'+file[8:]+'.h5')

Ypred1 = model1.predict(Xtest)
Ypred2 = model2.predict(Xtest)
Ypred3 = model3.predict(Xtest)

Vote1 = np.zeros_like(Ypred1)
Vote1[range(len(Vote1)),Ypred1.argmax(1)]=1
Vote2 = np.zeros_like(Ypred2)
Vote2[range(len(Vote2)),Ypred2.argmax(1)]=1
Vote3 = np.zeros_like(Ypred3)
Vote3[range(len(Vote3)),Ypred3.argmax(1)]=1

Vote = np.zeros_like(Vote1)

for i in range(Vote.shape[0]):
    if (np.array_equal(Vote2[i],Vote3[i])):
        Vote[i]=Vote2[i]
    else:
        Vote[i]=Vote1[i]

util.plot_confusion_matrix(Ytest,Vote,num_classes)