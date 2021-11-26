'''
3D convolutional neural network for class prediction

@author: Miguel Marcos
'''

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

path = 'D:\\Miguel\\out_flow\\'
file = 'out_flow_f8_mf500'

out_file = path+'results'+file[8:]+'.txt'
out_model = path+'model'+file[8:]+'.h5'
f = open(out_file,"w")

# ---------------------------------
# Parameter combinators

# Configure the vars and lists below 
# and the program will combine them 
# all to find the best combination!

# (Yeah brute force isn't very fancy
# or fast, but you gotta do what you
# gotta do :/  Also I have no clue
# of what to try at this point, so fuck
# it, I'm going full monke)
# ---------------------------------

# Number of iterations for averaging
num_it = 3

# Dimensions of kernel (only square size and depth)
# k_size must be lower than/equal to 10, and k_depth lower than/equal to 9
k_sizes = [3,4,5]
k_depths = [1,3,9]

# Activation function for the convolutional layers
activations = ['relu','tanh']

# Double convolution, yes or no (adds or discards the second Conv3D layer)
# The second kernel size will always be (3,3,3) unless k_depth is higher than 7
# in which case it doesn't fit, so it will be (3,3,1)
# We ain't got time for yet another loop to find another kernel size.
double_conv = [True,False]

# Add a pooling layer or not. If added, it will be [2,2,2] (half the output)
pooling = [True,False]

# Add a hidden linear layer or not after the convolution and pooling.
hidden = [True,False]

# Earlystopping patience (epochs without improvement)
patiences = [2]

# Dropout levels
dropouts = [0,0.15,0.3]

# ---------------------------------
# Main program

# Ok, here goes nothing
# ---------------------------------

print('Loading dataset...')
Xtrain, Xtest, Ytrain, Ytest, num_classes, classes = util.load_3d_data(path,file,img_dim,hog_dirs)
print('Training samples: ',Xtrain.shape[0])
print('Testing samples: ',Xtest.shape[0])

print('Commencing training loop')

f.write('Training loop for '+str(num_classes)+' classes:\n')
f.write(str(classes)+'\n')
f.write('Total training samples: '+str(Xtrain.shape[0])+'\n')
f.write('\n-Commencing loop-\n')

best_score = 0
best_config = ()

for size in k_sizes:
    for depth in k_depths:
        for act in activations:
            for dcv in double_conv:
                for pool in pooling:
                    for hid in hidden:
                        for pat in patiences:
        
                            # Stop training when validation error no longer decreases
                            earlystop=EarlyStopping(monitor='val_loss', patience=pat, 
                                                    verbose=0, mode='auto')
                        
                            for drop in dropouts:
                        
                                f.write('\n- - - - -\n')
                                f.write('Kernel: ('+str(size)+','+str(size)+','+str(depth)+')\n') 
                                f.write('Conv. Activ. f.: '+act)
                                f.write('DoubleConv: '+str(dcv)+'\n')
                                f.write('Patience: '+str(pat)+'\n')
                                f.write('Dropout: '+str(drop)+'\n')
                                f.write('Pooling: '+str(pool)+'\n')
                                f.write('Hidden-fully-connected: '+str(hid)+'\n')
                                f.write('- - - - -\n')
                                
                                config = (size,depth,dcv,pat,drop)
                                total_score = 0
                
                                for i in range(num_it):
                                    
                                    f.write('Iteration: '+str(i)+'\n')
                            
                                    (x_train,y_train),(x_val,y_val) = util.separate_3d_val_data(Xtrain,Ytrain,num_classes,img_dim,hog_dirs)
                                
                                    # convert class vectors to binary class matrices
                                    y_train = keras.utils.to_categorical(y_train, num_classes)
                                    y_val  = keras.utils.to_categorical(y_val,  num_classes)
                                    
                                    model = util.getUntrainedConv(size,depth,act,
                                                                  dcv,drop,pool,hid,
                                                                  input_shape,num_classes)
                                
                                    # Model training
                                    model.fit(x_train, y_train,
                                        batch_size=8,
                                        epochs=50,
                                        validation_split=0.2,
                                        callbacks=[earlystop],
                                        verbose=True)
                            
                                    # Model evaluation
                                    # train_score = model.evaluate(x_train, y_train, verbose=0)
                                    val_score = model.evaluate(x_val, y_val, verbose=0)
                                    # print('%s %2.2f%s' % ('Accuracy train: ', 100*train_score[1], '%' ))
                                    # print('%s %2.2f%s' % ('Accuracy test:  ', 100*test_score[1], '%'))
                                    
                                    f.write('\tValidation test Acc: '+str(100*val_score[1])+'%\n')
                                    
                                    total_score = total_score + val_score[1]
                                    
                                # end for  i
                                
                                av_score = total_score / num_it
                                f.write('\nAverage accuracy: '+str(100*av_score)+'%\n')
                                print('Finished testing model: '+str(config))
                                if av_score > best_score:
                                    best_score = av_score
                                    best_config = config
                                    print('Updated best model!')
                                    f.write('New best model!\n')
                        
                            # end for drop
                        # end for pat
                    #end for hid
                #end for pool
            # end for dcv
        #end for act
    # end for depth
# end for size

# Train the best model again

best_size,best_depth,best_dcv,best_pat,best_drop = best_config

best_model = util.getUntrainedConv(best_size,best_depth,best_dcv,best_pat,best_drop,input_shape,num_classes)

# Stop training when validation error no longer decreases
earlystop=EarlyStopping(monitor='val_loss', patience=best_pat, 
                        verbose=0, mode='auto')

history = best_model.fit(Xtrain, Ytrain,
                batch_size=4,
                epochs=100,
                validation_split=0.2,
                callbacks=[earlystop],
                verbose=True)

final_test = best_model.evaluate(Xtest,Ytest,verbose=0)
Ypred = best_model.predict(Xtest)

f.write('\tSelected best model: '+str(best_config)+'\n')
f.write('\tTest Acc: '+str(100*final_test[1])+'%\n')

util.plot_confusion_matrix(Ytest,Ypred,num_classes)
util.plot_history(history)

best_model.save(out_model)
        
f.close()