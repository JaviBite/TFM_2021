import numpy as np
import cv2
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

class FlowGenerator(Sequence) :
  
  def __init__(self, json_filename, labels, batch_size, dimension=250, padding=10, frames_sample=40, video_folder=None,
                    augmentation=False, balance=True, random_order=False, disbalance_factor = 30, verbose=False,
                    max_segments=5, flatten=False) :
    self.labels = labels
    self.frames_sample = frames_sample
    self.padding = padding
    self.dimension = dimension
    self.augmentation = augmentation
    self.disbalance_factor = disbalance_factor
    self.video_folder = video_folder
    self.flatten = flatten

    self.max_segments = max_segments

    if self.augmentation:
        self.batch_size = batch_size // 2
    else:
        self.batch_size = batch_size

    #Load json data 
    f1 = open(json_filename, encoding='utf-8')
    self.data = json.load(f1)

    # Set videos folder
    if self.video_folder is None:
        self.video_folder = self.data['config']['file']['loc_prefix']['1'][8:]

    # Create list of video fragments and rois
    video_data = {}
    self.regions = {}
    for value in self.data['metadata'].values():

        vid = int(value['vid'])

        # Temporal segment
        if len(value['xy']) == 0:
            action = str(value['av']['1'])

            #if word in action and vid not in video_ids:
            for word in self.labels:
                if re.search(word,action) is not None:

                    if vid not in video_data:
                        video_data[vid] = []

                    to_append = value
                    to_append['match'] = self.labels.index(word)
                    video_data[vid].append(to_append)
                    break

        # Region data
        else:

            if vid not in self.regions:
                self.regions[vid] = []

            self.regions[vid].append(value)

    # Short regions in chronological order
    for vid in self.regions.keys():
        self.regions[vid] = sorted(self.regions[vid], key=lambda d: d['z'][0], reverse=False)

    repVideos = dict(sorted(video_data.items(), reverse=False))

    # Create list of video fragments (look up table self.framgets[i])
    self.fragments = []
    for vid, data in zip(repVideos, repVideos.values()):

        videoPath = getVidPath(self.data, self.video_folder, str(vid)) 
        for elem in data:
            if len(elem['z']) == 2:

                # Temporal segment if the fragment duration allows it
                segments = int((elem['z'][1] - elem['z'][0]) / (self.frames_sample / FPS)) # 25 fps
                for i in range(min(segments,self.max_segments)):
                    self.fragments.append({
                        'vid': vid,
                        'vpath': videoPath,
                        'time': [elem['z'][0] + i * (self.frames_sample / FPS), elem['z'][0] + (i+1) * (self.frames_sample / FPS)],
                        'match': elem['match']
                    })

    if balance:
        if verbose:
            print("Classifing fragments in ", len(self.labels), "classes...")
        # Stack of fragments
        class_stacks = [[] for _ in range(len(self.labels))]

        for frag in self.fragments:
            class_id = frag['match']
            class_stacks[class_id].append(frag)

        if verbose:
            print("Stacks:")
            for ind in range(len(class_stacks)):
                print(self.labels[ind], " -> ", len(class_stacks[ind]))

        min_per_class = min([len(stack) for stack in class_stacks]) + self.disbalance_factor
        if verbose:
            print("Minimun + ", self.disbalance_factor, " = ", min_per_class)

        # Clean fragments
        self.fragments.clear()

        self.fragments = []
        # Add balanced fragments
        for ind in range(len(class_stacks)):
            stack = class_stacks[ind]
            self.fragments.extend(stack[:min(min_per_class,len(stack))])

        if verbose:
            print("Fragments len: ", len(self.fragments))

    if random_order:
        random.shuffle(self.fragments)

  def __len__(self) :
    return (np.ceil(len(self.fragments) / float(self.batch_size))).astype(np.int)

  def printClassesCount(self):
        classes_count = [0 for _ in range(len(self.labels))]
        for frag in self.fragments:
            classes_count[frag['match']] += 1

        print("Classes count:")
        for ind in range(len(classes_count)):
            print(self.labels[ind], " -> ", classes_count[ind])

  def setFragmentsSplit(self, split_ids):

      new_fragments = []
      for id in split_ids:
          new_fragments.append(self.fragments[id])
      
      self.fragments = new_fragments

  def get_splits(self, train_test=0.2, train_val=0.2):

        trainGenerator = copy.deepcopy(self)
        testGenerator = copy.deepcopy(self)
        valGenerator = copy.deepcopy(self)

        X_ids = []
        y = []
        for id, frag in enumerate(self.fragments):
            X_ids.append(id)
            y.append(frag['match'])

        Xtrain_ids, Xtest_ids, ytrain, _, = train_test_split(X_ids, y, test_size=train_test, random_state=RANDOM_STATE ,stratify=y)
        Xtrain_ids, Xval_ids, _, _, = train_test_split(Xtrain_ids, ytrain, test_size=train_val, random_state=RANDOM_STATE ,stratify=ytrain)

        trainGenerator.setFragmentsSplit(Xtrain_ids)
        testGenerator.setFragmentsSplit(Xtest_ids)
        valGenerator.setFragmentsSplit(Xval_ids)

        return trainGenerator, testGenerator, valGenerator
  
  # return the sequence, label one-hot econding
  # [[...], ..., [...]], [0,0...,1,0...] x batchsize
  def __getitem__(self, idx) :

    fragments = self.fragments[idx * self.batch_size : (idx + 1) * self.batch_size]

    batch_X = []
    batch_y = []

    for frag in fragments:

        # Load video
        vid = frag['vid']
        vpath = frag['vpath']

        cap = cv2.VideoCapture(vpath)
        fps = FPS

        init_frame = int(frag['time'][0] * fps)
        final_frame = int(frag['time'][1] * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

        # Get roi
        possible_rois = closest_rois(self.regions[vid], init_frame / fps)
        roi = getMostFlowRoi(possible_rois, cap, init_frame, int(init_frame+(self.frames_sample/2)),padding=self.padding, visualize=VIS)

        if roi is not None:
            x1, y1, x2, y2 = roi
            roi_window = [slice(y1,y1+y2), slice(x1,x1+x2)]
        else:
            print("No ROI")
            continue

        # Process the video and get the flow sequence
        
        cap.set(cv2.CAP_PROP_POS_FRAMES,init_frame)
        ret, frame = cap.read()
        frameflow1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        last_gray_frames = [None, None]
        last_gray_frames_flip = [None, None]

        #Initial flow        
        last_gray_frames[1] = cv2.resize(frameflow1[tuple(roi_window)],(self.dimension,self.dimension))
        last_gray_frames_flip[1] = cv2.resize(cv2.flip(frameflow1[tuple(roi_window)],1),(self.dimension,self.dimension))
    
        sequence = []
        sequence_aug = []
        frame_i = init_frame + 1 # Inital flow

        while (cap.isOpened() and frame_i <= final_frame):

            # Capture frame-by-frame
            ret, frame = cap.read()
                
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_frame = cv2.resize(gray_frame[tuple(roi_window)],(self.dimension,self.dimension))

            # Update Last Frames
            last_gray_frames[0] = last_gray_frames[1]
            last_gray_frames[1] = roi_frame
        
            #Calcular flujo optico
            flowFB = cv2.calcOpticalFlowFarneback(last_gray_frames[0], last_gray_frames[1], 
                            None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            if self.augmentation:
                roi_flip_frame = cv2.flip(roi_frame, 1)
                last_gray_frames_flip[0] = last_gray_frames_flip[1]
                last_gray_frames_flip[1] = roi_flip_frame
                flowFB_flip = cv2.calcOpticalFlowFarneback(last_gray_frames_flip[0], last_gray_frames_flip[1], 
                            None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            # Visualization
            if VIS:
                img_roi = frame[tuple(roi_window)]
                img_roi = cv2.resize(img_roi,(self.dimension,self.dimension))
                cv2.imshow("ROI", img_roi)
                cv2.imshow("Image", frame)
                cv2.waitKey(100)
            
            # Put the flow into the sequence
            if VIS:
                _,  flow_HSV2 = draw_hsv(flowFB)
                cv2.imshow('flow', flow_HSV2)  
            
            if self.flatten:
                flowFB = flowFB.flatten()
            sequence.append(flowFB) 

            if self.augmentation:
                if self.flatten:
                    flowFB_flip = flowFB_flip.flatten()
                sequence_aug.append(flowFB_flip) 

            frame_i =  frame_i + 1

        # End framgnet processing
        #print("len seq: ", len(sequence))
        sequence = sequence[:self.frames_sample]
        if self.augmentation:
            sequence_aug = sequence_aug[:self.frames_sample]
            
        # TODO normalize flow along the sequence

        # Append the sequence to the batch
        batch_X.append(sequence)
        batch_y.append(frag['match'])

        if self.augmentation:
            batch_X.append(sequence_aug)
            batch_y.append(frag['match'])

    batch_y = to_categorical(batch_y, num_classes=len(self.labels))
    batch_X, batch_y = np.array(batch_X), np.array(batch_y)
    #print(batch_X.shape, batch_y.shape)

    return batch_X, batch_y