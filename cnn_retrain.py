# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 19:27:09 2021

@author: Miguel
"""

import keras
from keras.callbacks import EarlyStopping

import bsh_nn_utils as util

import json

import numpy as np

# ---------------------------------
# Data metrics for input and output
# ---------------------------------

img_dim = 15
hog_dirs = 9
input_shape = (img_dim,img_dim,hog_dirs) # Number of inputs for the neural network

path = 'D:\\GI Lab\\'
file = 'out_flow_f40_mf500'

out_path = 'D:\\GI Lab\\Pruebas_cnn_2\\'

out_model_path = out_path+'models\\'
model_path = out_model_path+'model_'
model_name = 'f8_mf5000_retrained'
model_endname = '_retrained.h5'

# json_path = out_path+'results_'+model_name+'.json'

# Number of models to be retrained
num_models = 14

# ---------------------------------
# Main program

# Ok, here goes nothing, again
# ---------------------------------

print('Loading dataset...')
Xtrain, Xtest, Ytrain, Ytest, num_classes, classes = util.load_3d_flat_w_data(path,file,img_dim,hog_dirs)
print('Training samples shape: ',Xtrain.shape)
print('Tag shape: ',Ytrain.shape)

# json_f = open(json_path)
# json_data = json.load(json_f)
# json_f.close()

print('Commencing training loop')

# Number of iterations for averaging
# ( the 'k' value for  k-fold cross-validation )
num_it = 5

# Models with less accuracy than this value will be discarded
score_threshold = 0.45

kf_x_train, kf_x_val, kf_y_train, kf_y_val = util.kfold(Xtrain,Ytrain,num_classes,img_dim,hog_dirs,num_it)

Ytrain = keras.utils.to_categorical(Ytrain, num_classes)
Ytest = keras.utils.to_categorical(Ytest, num_classes)

# Stop training when validation error no longer decreases
earlystop=EarlyStopping(monitor='val_loss', patience=5, 
                        verbose=0, mode='auto')

for i in range(num_models):
    
    total_score = 0
    
    for it in range(num_it):
    
        model = keras.models.load_model(model_path+str(i+1)+'_'+model_name+'.h5')
        
        print('Iteration: '+str(it)+'...')
                                
        x_train = kf_x_train[it]
        x_val = kf_x_val[it]
        y_train = kf_y_train[it]
        y_val = kf_y_val[it]
    
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train,num_classes)
        y_val  = keras.utils.to_categorical(y_val,num_classes)
        
    
        # Model training
        model.fit(x_train, y_train,
                  batch_size=8,
                  epochs=50,
                  validation_split=0.15,
                  callbacks=[earlystop],
                  verbose=False)

        # Model evaluation
        val_score = model.evaluate(x_val, y_val, verbose=0)
        print('%s %2.2f%s' % ('Accuracy test:  ', 100*val_score[1], '%'))
        total_score = total_score + val_score[1]
        
    # end for  it
    
    av_score = total_score / num_it
    print('Finished testing model: '+str(i))
    print('Average accuracy: '+str(100*av_score)+'%')
    
    # og_score = json_data[str(i+1)]["Avg. acc."]
    
    if av_score > score_threshold: # and av_score>og_score:
        
        print('Model accepted, retraining.')
    
        model = keras.models.load_model(model_path+str(i+1)+'_'+model_name+'.h5')
        
        model.fit(Xtrain,Ytrain,
                batch_size=4,
                epochs=100,
                validation_split=0.15,
                callbacks=[earlystop],
                verbose=True)
        
        out_model = model_path+str(i+1)+'_'+model_name+model_endname
        model.save(out_model)
        
        print('Saving model')
        
    else:
        print('Discarded')
