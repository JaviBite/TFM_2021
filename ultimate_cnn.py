# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:19:16 2021

This program trains and/or evaluates convolutional neural networks for
action classification using HOGs extracted from cooking sample videos.

It performs a loop over several training parameters in order to find
as many models over a certain accuracy threshold as possible, then
performs a discarding sequence to preserve variance, and finally
applies an agreement algorythm between all the remaining models.
The training and discarding stages may be skipped. Already trained
models can be provided via argument parsing.

This script and its imported utilities are intended to be 
used by BSH for action recognition with specialized hardware.

@author: Miguel Marcos - Graphics and Imaging Lab intern
"""

# IMPORTS #

# For model definition, training and prediction #
import keras

# For data reading and model evaluation #
import bsh_nn_utils_def as util

# For json data reading and writing #
import json

# For data manipulation #
import numpy as np

# For argument parsing #
import argparse

# END IMPORTS #

# ----------------------------- #

# ARGUMENT PARSING #

# Default arguments

THR = 0.5 # Score threshold for model discarding
MINM = 10 # Minimum number of models to keep after best-friend discarding

# Argument parser

parser = argparse.ArgumentParser()

parser.add_argument("data_path",type=str,help="Path and common name of the stored npz and json files")
parser.add_argument("model_path",type=str,help="Path and common name to store the models and results")
parser.add_argument("-w","--window",action="store_true",help="If set,frame windows will be taken on account")
parser.add_argument("-nt","--notrain",action="store_true",help="If set,training is skipped and models are loaded from model_path")
parser.add_argument("-nd","--nodisc",action="store_true",help="If set, model discarding is skipped")
parser.add_argument("-th","--threshold",type=float,default=THR,help="Accuracy threshold for models")
parser.add_argument("-m","--minmodels",type=int,default=MINM,help="Minimum number of models to keep from all the obtained")

args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path
window = args.window
notrain = args.notrain
nodisc = args.nodisc
sc_thresh = args.threshold
minmodels = args.minmodels

# END ARGUMENT PARSING #

# ---------------------------- #

# DATA LOADING #

print('Loading dataset...')
Xtrain, Xtest, Ytrain, Ytest, num_classes, classes, windows = util.load_data(data_path,window)
print('Training dataset shape: ',Xtrain.shape)
print('Testing dataset shape: ',Xtest.shape)

# Half the test samples are separated for model discarding
# If model discarding is skipped, the full testing dataset is used.
num_test_samples = Xtest.shape[0]
Xtest_md = Xtest[:(num_test_samples//2)] 
Ytest_md = Ytest[:(num_test_samples//2)] # For model discarding
Ytest_md = keras.utils.to_categorical(Ytest_md)
Xtest_mp = Xtest[(num_test_samples//2):] 
Ytest_mp = Ytest[(num_test_samples//2):] # For model prediction
Ytest_md = keras.utils.to_categorical(Ytest_mp)
    
input_shape = (Xtrain.shape[1],Xtrain.shape[2],Xtrain.shape[3])

# END DATA LOADING #

# ---------------------------- #

# MODEL TRAINING #

models = []     # An array for models
json_data = {}  # A dictionary for model data

if not notrain:
    
    # Train new models
    
    json_data, models = util.trainloop(Xtrain,Ytrain,num_classes,sc_thresh,
                                      input_shape,model_path)
    
    nummodels = len(models)

else: # if notrain

    # Load already trained models
    
    json_f = open(model_path+model_name+".json")
    json_data = json.load(json_f)
    json_f.close()
    
    nummodels = 0

    for j in json_data:
    
        model_fullname = model_path+"_model_"+j+".h5"
        
        models.append(keras.models.load_model(model_fullname))
        nummodels += 1
        
# end if
    
# END MODEL TRAINING #

# ---------------------------- #

# MODEL DISCARDING #

Valid = [True]*nummodels

if not nodisc:

    # Begin model discarding
    
    votes_md = [np.zeros_like(Ytest_md)]
    
    # Every model predicts classes for half the test data
    
    for i in range(nummodels):
        
        model_vote = models[i].predict(Xtest_md)
        votes_md = np.append(votes_md,[model_vote],axis=0)
        
    # end for i
    
    votes_md = votes_md[1:]
    discarded_models = 0
    
    for i in range(nummodels):
        
        # For every valid model, the one with the most similar vote is found
    
        if not Valid[i]: continue
    
        min_norm = np.inf
        best_friend = i
        
        i_vote = votes_md[i]
        
        for j in range(i+1,nummodels):
            
            if not Valid[j]: continue
        
            j_vote = votes_md[j]
            
            norm = np.linalg.norm(i_vote-j_vote)
            
            if norm < min_norm:
                best_friend = j
                min_norm = norm
                
            # end if
        # end for j
        
        # And the one with the lowest accuracy is discarded
        
        if best_friend != i :
            
            i_acc = util.getAccuracy(Ytest_md,votes_md[i])
            j_acc = util.getAccuracy(Ytest_md,votes_md[best_friend])
            
            if i_acc < j_acc : 
                Valid[i] = False 
            else : 
                Valid[best_friend] = False 
            # end if
            
            discarded_models += 1
        
        # end if
        
        if (nummodels - discarded_models) <= minmodels: break
        
    # end for i
# end if

# END MODEL DISCARDING #

# ---------------------------- #

# MODEL SAVING #
    
# Only save models if they are newly trained

if not notrain:
    
    for i in range(nummodels):
        
        if Valid[i]:
            
            out_model = model_path+str(i+1)+".h5"
            models[i].save(out_model)
    
        else: # if not Valid[i]
            
            del json_data[i+1]
    
        # end if
    # end for
    
    json.
    
# end if

# END MODEL SAVING #

# ---------------------------- #

# PREDICTION #

if nodisc:
    
    # If discarding was skipped, the test dataset is used entirely
    
    Xtest_mp = np.append(Xtest_mp,Xtest_md,axis=0)
    Ytest_mp = np.append(Ytest_mp,Ytest_md,axis=0)
    
# end if

model_votes = [np.zeros(num_classes)]*nummodels
model_accs = np.zeros(nummodels)
model_top2accs = np.zeros(nummodels)
vote_urn = np.zeros(num_classes)

for w in range(windows):
    
    # If -w is unset, only one iteration is done, as windows=1
    
    Xtest_mp_w = Xtest_mp[:,w]
    
    for i in range(nummodels):
        
        if not Valid[i]: continue
    
        model = models[i]
        
        Ypred = model.predict(Xtest_mp_w)
        
        # Vote system: Everyone sums their predicted chances for each class
        model_vote = Ypred
        model_votes[i] = model_vote
        model_accs[i] = util.getAccuracy(Ytest_mp, model_vote)
        model_top2accs[i] = util.getTop2Acc(Ytest_mp, model_vote)
        vote_urn += model_vote
        
    # end for i
    
    print("Window:",w)
    print("---Voting Accuracy:",util.getAccuracy(Ytest_mp, vote_urn))
    print("---Average Accuracy:",np.round(np.mean(model_accs),3))
    print("---Voting Top2Accuracy:",util.getTop2Acc(Ytest_mp, vote_urn))
    print("---Average Top2Accuracy:",np.round(np.mean(model_top2accs),3))
    
#end for w


