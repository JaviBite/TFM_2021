# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:43:25 2021

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
num_w = 3
input_shape = (img_dim,img_dim,hog_dirs) # Number of inputs for the neural network

path = 'D:\\GI Lab\\'
file = 'out_flow_f24_mf1000_w_r'

directory = 'D:\\GI Lab\\Pruebas_cnn_2\\'
json_path = directory+'results_'+file[9:]+'.json'
model_path = directory+'models\\model_'
model_endname = '_f8_mf5000_retrained_retrained.h5'

# ---------------------------------
# Data reading
# ---------------------------------

Xtrain, Xtest, Ytrain, Ytest, num_classes, classes = util.load_3d_w_data(path,file,num_w,img_dim,hog_dirs)

Ytrain = keras.utils.to_categorical(Ytrain, num_classes)
Ytest = keras.utils.to_categorical(Ytest, num_classes)

num_test_samples = Xtest.shape[0]
windows = Xtest.shape[1]
Xtest_md = Xtest[:(num_test_samples//2)]
num_md_samples = Xtest_md.shape[0]
Xtest_md = np.reshape(Xtest_md,(num_md_samples*windows,img_dim,img_dim,hog_dirs)) 
Ytest_md = Ytest[:(num_test_samples//2)] # For model discarding
Ytest_md = np.array([val for val in Ytest_md for _ in range(windows)])
Xtest_mp = Xtest[(num_test_samples//2):] 
Ytest_mp = Ytest[(num_test_samples//2):] # For model prediction

# json_f = open(json_path)
# json_data = json.load(json_f)
# json_f.close()

num_models = 14
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
# #end for

Votes_md = Votes_md[1:]
Valid = [True]*num_models
discarded_models = 0

# for i in range(num_models):

#     if num_models-discarded_models <= num_models//2 : break
    
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

# ---------------------------------

Votes = [np.zeros_like(Ytest_mp)]*num_w
RawVotes = [np.zeros_like(Ytest_mp)]*num_w
AvProg = 0

for i in range(num_models):
    
    if not Valid[i]: continue
    
    print("Model: ",i+1)

    model_name = model_path+str(i+1)+model_endname
    
    model = keras.models.load_model(model_name)
    MyVotes = [np.zeros_like(Ytest)]*num_w
        
    for j in range(num_w):
        
        window = Xtest_mp[:,j]
    
        model_vote = model.predict(window)
        
        if j == 0:
            MyVotes[0] = model_vote
        else:
            MyVotes[j] = MyVotes[j-1]*j/(j+1) + model_vote/(j+1)
        
        Votes[j] = Votes[j] + MyVotes[j]
        RawVotes[j] = RawVotes[j] + model_vote
        
    # end for
    
    ini_acc = util.getAccuracy(Ytest_mp, MyVotes[0])
    
    final_acc = util.getAccuracy(Ytest_mp, MyVotes[num_w-1])
    
    print(ini_acc,"---->", final_acc)
    
    AvProg += (final_acc-ini_acc)
        
# end for


util.plot_confusion_matrix(Ytest_mp,Votes[num_w-1],num_classes)

print()
print('# of models over the threshold: '+str(num_models))
print('# of models discarded by similarity: '+str(discarded_models))
print('Average per-model acc. progression:',AvProg/(num_models-discarded_models))
print('Agreement accuracy progression: ')
print(str(util.getAccuracy(Ytest_mp,Votes[0])),"--->",str(util.getAccuracy(Ytest_mp,Votes[num_w-1])))
print('F1scores per class: '+str(util.getF1scores(Ytest_mp,Votes[num_w-1])))


