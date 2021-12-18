'''
3D convolutional neural network for class prediction

@author: Miguel Marcos
'''

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
file = 'out_flow_f24_mf1000_w_r'

out_path = 'D:\\GI Lab\\Pruebas_cnn_2\\'
out_file_path = out_path+'results_'+file[9:]+'.json'

num_model = 1
out_model_path = out_path+'models\\'
model_path = out_model_path+'model_'
model_endname = '_'+file[9:]+'.h5'

# Models with less accuracy than this value will be discarded
score_threshold = 0.5

# ---------------------------------
# Parameter combinators

# Configure the vars and lists below 
# and the program will combine them 
# all to find the best combination!

# (Yeah brute force isn't very fancy
# or fast, but you gotta do what you
# gotta do :/ )
# ---------------------------------

# Number of iterations for averaging
# ( the 'k' value for  k-fold cross-validation )
num_it = 4

# Dimensions of kernel (only square kernels will be tested)
# Beware bigger k-sizes depending on the number of conv. layers
k_sizes = [3]

# Activation function for the convolutional layers
activations = ['tanh','relu']

# Number of convolutional layers. Second layer onward will always have a
# 3 wide, 3 tall kernel.
num_conv = [1,2,3]

# Add a pooling layer or not. If added, it will be (2,2) 
# (halves the input size in every dimension)
pooling = [True,False]

# Add a hidden linear layer or not after the convolution and pooling.
hidden = [True,False]

# Earlystopping patience (epochs without improvement)
patiences = [2,10]

# Dropout levels
dropouts = [0.1,0.3]

# ---------------------------------
# Main program

# Ok, here goes nothing
# ---------------------------------

print('Loading dataset...')
Xtrain, Xtest, Ytrain, Ytest, num_classes, classes = util.load_3d_flat_w_data(path,file,img_dim,hog_dirs)
print('Training samples shape: ',Xtrain.shape)
print('Tag shape: ',Ytrain.shape)

print('Commencing training loop')

out_json = {}


kf_x_train, kf_x_val, kf_y_train, kf_y_val = util.kfold(Xtrain,Ytrain,num_classes,img_dim,hog_dirs,num_it)

Ytrain = keras.utils.to_categorical(Ytrain, num_classes)
Ytest = keras.utils.to_categorical(Ytest, num_classes)

for size in k_sizes:
    for act in activations:
        for conv in num_conv:
            for pool in pooling:
                for hid in hidden:
                    for pat in patiences:
    
                        # Stop training when validation error no longer decreases
                        earlystop=EarlyStopping(monitor='val_loss', patience=pat, 
                                                verbose=0, mode='auto')
                    
                        for drop in dropouts:
                            
                            config = ((size,size),act,conv,pat,drop,pool,hid)
                            total_score = 0
            
                            for i in range(num_it):
                                
                                print('Iteration: '+str(i)+'...')
                                
                                x_train = kf_x_train[i]
                                x_val = kf_x_val[i]
                                y_train = kf_y_train[i]
                                y_val = kf_y_val[i]
                            
                                # convert class vectors to binary class matrices
                                y_train = keras.utils.to_categorical(y_train,num_classes)
                                y_val  = keras.utils.to_categorical(y_val,num_classes)
                                
                                model = util.getUntrained2DConv(size,act,
                                                                conv,drop,pool,hid,
                                                                input_shape,num_classes)
                            
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
                                
                            # end for  i
                            
                            av_score = total_score / num_it
                            print('Finished testing model: '+str(config))
                            print('Average accuracy: '+str(100*av_score)+'%')
                            
                            if av_score > score_threshold:
                                
                                out_json[num_model] = {'Kernel': (size,size),
                                                       'Act. f.': act,
                                                       'Num. Conv.': conv,
                                                       'Patience': pat,
                                                       'Dropout': drop,
                                                       'Pooling': pool,
                                                       'Hidden': hid,
                                                       'Avg. acc.': av_score,
                                                       }
                                
                                print('Model accepted, retraining.')
                                
                                model = util.getUntrained2DConv(size,act,
                                                              conv,drop,pool,hid,
                                                              input_shape,num_classes)
                                
                                model.fit(Xtrain,Ytrain,
                                        batch_size=4,
                                        epochs=100,
                                        validation_split=0.15,
                                        callbacks=[earlystop],
                                        verbose=True)
                                
                                out_model = model_path+str(num_model)+model_endname
                                model.save(out_model)
                                num_model = num_model+1
                                
                                print('Saving model')
                                
                            else:
                                print('Discarded')
                    
                        # end for drop
                    # end for pat
                #end for hid
            #end for pool
        # end for dcv
    #end for act
# end for size

out_json_file = open(out_file_path,'w')
json.dump(out_json,out_json_file,indent=1)
out_json_file.close()
