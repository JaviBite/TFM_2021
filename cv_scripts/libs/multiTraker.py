
import numpy as np
import time, math
import os, json, cv2, random, torch, sys
from glob import glob
import torch.nn.functional as F
from statistics import median, mean

# pysot packages
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.tracker_builder import build_tracker


from detectron2.structures import Boxes, Instances, pairwise_ioa

class SiamRPNMultiTracker():
    def __init__(self, model_path):
            self.trackers = {}
            self.next_id = 0
            self.model_path = model_path
            self.num = 0
            self.device = torch.device('cuda' if cfg.CUDA else 'cpu')
            self.vel_update = 25
            self.vel_update_count = self.vel_update
            
    def addTracker(self, frame, init_rect, box, nframe, info):
        new_model = ModelBuilder()
        new_model.load_state_dict(torch.load(self.model_path,
            map_location=lambda storage, loc: storage.cpu()))
        new_model.eval().to(self.device)
    
        new_tracker = build_tracker(new_model)                     
        new_tracker.init(frame, init_rect)
        self.trackers[self.next_id] = {'tracker': new_tracker, 'count': 0, 'box': box, 'obox': box, 'oframe': nframe, \
                            'info': info, 'bad_frames': 0, 'last_score': 1, 'jumps': 0, 'vel': 0, 'dists': {'list':[5] * 1, 'next_id':0, 'last_pos':[0,0]} }
        self.next_id = self.next_id + 1
        self.num = self.num + 1
    
    def update(self, id, frame, init_rect):        
        self.trackers[id]['tracker'].init(frame, init_rect)
                            
    def removeTraker(self, id):
        del self.trackers[id]
        self.num = self.num - 1
    
    def track(self, frame):
        ret = {} # dict: {id: [box, score]}
        for id in self.trackers:
            tracker = self.trackers[id]['tracker']
            outs = tracker.track(frame)
            
            # Get boxes
            bbox = list(map(int, outs['bbox']))
            score = outs['best_score']
            
            self.trackers[id]['last_score'] = score
            
            if score < 0.8:
                self.trackers[id]['bad_frames'] += 1
            #elif score > 0.9:
                #self.trackers[id]['bad_frames'] -= 1
                         
            if self.trackers[id]['bad_frames'] < 0:
                self.trackers[id]['bad_frames'] = 0
                    
            
            detBox = Boxes(torch.tensor([[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]]))
            detBox = detBox.to(self.device)
            
            oldBox = self.trackers[id]['box']
            
            dist = (oldBox.get_centers() - detBox.get_centers())
            dist = np.linalg.norm(dist.cpu().numpy())
            
            if self.vel_update_count <= 0:
                lastPos = self.trackers[id]['dists']['last_pos']
                nowPos = detBox.get_centers().cpu().numpy()
                distVel = np.linalg.norm(lastPos - nowPos)
                
                toadd = self.trackers[id]['dists']['next_id']
                self.trackers[id]['dists']['list'][0] = distVel
                self.trackers[id]['dists']['next_id'] = (toadd + 1) % 1
                self.trackers[id]['vel'] = self.trackers[id]['dists']['list'][0]#median(self.trackers[id]['dists']['list'])
                self.vel_update_count = self.vel_update
                
                self.trackers[id]['dists']['last_pos'] = nowPos
            else:
                self.vel_update_count -= 1
            
            
            
            if dist > 100:
                self.trackers[id]['jumps'] += 1
                self.trackers[id]['bad_frames'] += 5
                
            # Check areas
            oBox =  self.trackers[id]['obox']
            max_area = max(oldBox.area(), detBox.area())
            min_area = min(oldBox.area(), detBox.area())
            #print(min_area, max_area, min_area/max_area)
            if (min_area/max_area) < 0.5:
                self.trackers[id]['bad_frames'] += 10               
                                
            self.trackers[id]['box'] = detBox
            ret[id] = [bbox, score, dist]
            
        return ret
        
    def getBox(self, id):
        return self.trackers[id]['box']
        
    def getInfo(self, id):
        return self.trackers[id]['info']
    
    def getBadFrames(self, id):
        return self.trackers[id]['bad_frames']
    
    def getVelocity(self, id):
        return self.trackers[id]['vel']
    
    def addBadFrames(self, id, count = 1):
        self.trackers[id]['bad_frames'] += count
    
    def checkDups(self):
        for id in self.trackers:
            box1 = self.trackers[id]['box']
            class1 = self.trackers[id]['info']['class_id']
            for id2 in self.trackers:
                if (id < id2):
                    box2 = self.trackers[id2]['box']
                    class2 = self.trackers[id2]['info']['class_id']
                    
                    iou = pairwise_ioa(box1, box2)
                    
                    max_area = max(box1.area(), box2.area())
                    min_area = min(box1.area(), box2.area())
                    areaRatio = min_area/max_area
                        
                    if iou >= 0.5 and areaRatio > 0.5 and (class1 == class2 or (class1 == 41 and class2 == 45) or (class1 == 45 and class2 == 41)):
                        
                        # Check if tracked jumped to another
                        if self.trackers[id]['jumps'] > self.trackers[id2]['jumps']:
                            self.trackers[id]['bad_frames'] += 60
                        elif self.trackers[id]['jumps'] < self.trackers[id2]['jumps']:
                            self.trackers[id2]['bad_frames'] += 60
                        
                        # Check bad frames
                        elif self.trackers[id]['bad_frames'] > self.trackers[id2]['bad_frames']:
                            self.trackers[id]['bad_frames'] += 60
                        elif self.trackers[id]['bad_frames'] < self.trackers[id2]['bad_frames']:
                            self.trackers[id2]['bad_frames'] += 60
                            
                        # Check score
                        elif self.trackers[id]['last_score'] > self.trackers[id2]['last_score']:
                            self.trackers[id2]['bad_frames'] += 60
                        else:
                            self.trackers[id]['bad_frames'] += 60
                            
    def removeBadTrackers(self):
        toRemove = []
        listRemoved = []
        for id in self.trackers:
            if self.trackers[id]['bad_frames'] > 30:
                toRemove.append(id)   
                listRemoved.append( self.trackers[id]['info'])
        
        for id in toRemove:
            self.removeTraker(id)
            
        return listRemoved
            
    
    def count(self):
        return self.num
        
    def __len__(self):
        return self.num