# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 00:39:31 2021

@author: Miguel
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta

import bsh_nn_utils as util


# ---------------------------------
# Data metrics for input and output
# ---------------------------------

img_dim = 15
hog_dirs = 9
input_shape = (img_dim,img_dim,hog_dirs,1) # Number of inputs for the neural network

path = 'D:\\GI Lab\\Pruebas_acc_v2\\'
file = 'out_flow_f8_mf500'
out_model = path+'model_3'+file[8:]+'.h5'

# Stop training when validation error no longer decreases
earlystop=EarlyStopping(monitor='val_loss', patience=2,verbose=0, mode='auto')

Xtrain, Xtest, Ytrain, Ytest, num_classes, classes = util.load_3d_data(path,file,img_dim,hog_dirs)

Ytrain = keras.utils.to_categorical(Ytrain, num_classes)
Ytest = keras.utils.to_categorical(Ytest, num_classes)

model = util.getUntrainedConv(3,9,'tanh',True,0,True,True,input_shape,num_classes)

history = model.fit(Xtrain, Ytrain,
                batch_size=4,
                epochs=100,
                validation_split=0.2,
                callbacks=[earlystop],
                verbose=True)

final_test = model.evaluate(Xtest,Ytest,verbose=0)
Ypred = model.predict(Xtest)

util.plot_confusion_matrix(Ytest,Ypred,num_classes)
util.plot_history(history)

#model.save(out_model)