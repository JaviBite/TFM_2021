# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 09:44:33 2021

@author: Miguel
"""

import keras

import bsh_nn_utils as util
import numpy as np

import json

# ---------------------------------
# Data metrics for input and output
# ---------------------------------

img_dim = 15
hog_dirs = 9
input_shape = (img_dim,img_dim,hog_dirs) # Number of inputs for the neural network

path = 'D:\\GI Lab\\'
file = 'out_flow_f8_mf5000'

directory = 'D:\\GI Lab\\Pruebas_cnn_2\\'
json_path = directory+'results_'+file[9:]+'.json'
model_path = directory+'models\\model_'
model_endname = '_f8_mf5000_retrained.h5'

# ---------------------------------
# Data reading
# ---------------------------------

Xtrain, Xtest, Ytrain, Ytest, num_classes, classes = util.load_3d_data(path,file,img_dim,hog_dirs)

Ytrain = keras.utils.to_categorical(Ytrain, num_classes)
Ytest = keras.utils.to_categorical(Ytest, num_classes)

num_test_samples = Xtest.shape[0]
Xtest_md = Xtest[:(num_test_samples//2)] 
Ytest_md = Ytest[:(num_test_samples//2)] # For model discarding
Xtest_mp = Xtest[(num_test_samples//2):] 
Ytest_mp = Ytest[(num_test_samples//2):] # For model prediction

json_f = open(json_path)
json_data = json.load(json_f)
json_f.close()

num_models = 13
min_models = 10
Votes_md = [np.zeros_like(Ytest_md)]

# ---------------------------------
# Model discarding
# Every model casts its vote for 
# half the test data samples.
# For each model, the one with most
# similar votes is selected, and the
# one of them with less accuracy is discarded.
# ---------------------------------

# print('Beginning best-friend-discarding')

# for i in range(num_models):
    
#     model_name = model_path+str(i+1)+model_endname
    
#     model = keras.models.load_model(model_name)
    
#     Ypred = model.predict(Xtest_md)
    
#     # Vote system: Everyone sums their predicted chances for each class
#     model_vote = Ypred
    
#     Votes_md = np.append(Votes_md,[model_vote],axis=0)
# # end for

Votes_md = Votes_md[1:]
Valid = [True]*num_models
discarded_models = 0

# for i in range(num_models):
    
#     if discarded_models >= num_models-min_models : break
    
#     if not Valid[i]: continue

#     min_norm = np.inf
#     best_friend = i
    
#     i_vote = Votes_md[i]
    
#     for j in range(i+1,num_models):
        
#         if not Valid[j]: continue
    
#         j_vote = Votes_md[j]
        
#         norm = np.linalg.norm(i_vote-j_vote)
        
#         if norm < min_norm:
#             best_friend = j
#             min_norm = norm
#         # end if
#     # end for
    
#     if best_friend != i :
        
#         i_acc = util.getAccuracy(Ytest_md,Votes_md[i])
#         j_acc = util.getAccuracy(Ytest_md,Votes_md[best_friend])
        
#         if i_acc < j_acc : 
#             Valid[i] = False 
#         else : 
#             Valid[best_friend] = False 
        
#         discarded_models += 1
    
#     # end if
# # end for

# print('Finished')
print('Beginning final test')

Votes = np.zeros_like(Ytest_mp)
max_acc = 0

for i in range(num_models):
    
    if not Valid[i]: continue

    model_name = model_path+str(i+1)+model_endname
    
    model = keras.models.load_model(model_name)
    
    # model_val_acc = json_data[str(i+1)]['Avg. acc.']
    # if model_val_acc > max_acc:
    #     max_acc = model_val_acc
    
    Ypred = model.predict(Xtest_mp)
    
    # Vote system: Everyone sums their predicted chances for each class
    model_vote = Ypred
    
    print("Model",i+1,": Acc",util.getAccuracy(Ytest_mp, Ypred))
    
    Votes = Votes+model_vote
# end for

util.plot_confusion_matrix(Ytest_mp,Votes,num_classes)
print('F1scores per class: '+str(util.getF1scores(Ytest_mp,Votes)))
print('Global accuracy: '+str(util.getAccuracy(Ytest_mp,Votes)))
print('Highest single-model validation accuracy: '+str(np.round(max_acc,3)))
print('# of models over the threshold: '+str(num_models))
print('# of models discarded by similarity: '+str(discarded_models))


