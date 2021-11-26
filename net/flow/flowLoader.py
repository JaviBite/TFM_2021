import numpy as np
import cv2, os
import json
import re
import random
import sys
import copy
from keras.utils.all_utils import to_categorical, Sequence

sys.path.append("../..")

from json_export_flow import getMostFlowRoi, closest_rois
from cv_scripts.flow_hog import draw_hsv
from sklearn.model_selection import train_test_split

VIS = False
FPS = 25
RANDOM_STATE = 42

def getVidPath(jsondata, localpath, vid):
    namepath = jsondata['file'][vid]['fname']
    return localpath + namepath

class FlowLoader(Sequence) :
  
  def __init__(self, flow_dir, labels, batch_size=4, random_order=False, verbose=False, flatten=False, metafile=None):
    self.flatten = flatten
    self.flow_dir = os.path.normpath(flow_dir)
    self.batch_size = batch_size
    self.data = []
    self.labels = labels

    for (dirpath, dirnames, filenames) in os.walk(self.flow_dir):
        for file in filenames:
            if file.endswith('.npz'):
                class_label = dirpath.split("\\")[-1]
                one_hot_label = to_categorical(self.labels.index(class_label), num_classes=len(self.labels))
                self.data += [[os.path.join(dirpath, file), one_hot_label]]

    if random_order:
        random.shuffle(self.data)

  def __len__(self) :
    return (np.ceil(len(self.data) / float(self.batch_size))).astype(np.int)

  def printClassesCount(self):
        classes_count = [0 for _ in range(len(self.labels))]
        for frag in self.fragments:
            classes_count[frag['match']] += 1

        print("Classes count:")
        for ind in range(len(classes_count)):
            print(self.labels[ind], " -> ", classes_count[ind])

  # return the sequence, label one-hot econding
  # [[...], ..., [...]], [0,0...,1,0...] x batchsize
  def __getitem__(self, idx) :

    batch_X = []
    batch_y = []

    for idx in range(idx * self.batch_size, (idx + 1) * self.batch_size):
        file, y = self.data[idx]

        x = np.load(file, allow_pickle=True)['a']
        if self.flatten:
            x = x.flatten()
        batch_X.append(x)
        batch_y.append(y)
       

    batch_X, batch_y = np.array(batch_X), np.array(batch_y)
    batch_X = batch_X.reshape(self.batch_size, batch_X.shape[2], batch_X.shape[3],batch_X.shape[4],batch_X.shape[5])

    return batch_X, batch_y