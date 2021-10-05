"""
OPTICAL FLOW

Created on Mon Feb 24 11:54:23 2020

@author: https://gist.github.com/FingerRec/eba088d6d7a50e17c875d74684ec2849
"""


from libs import mi_hog



import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import scipy.io

from pot_det import detect_pots

from cv2 import __version__
print(__version__)

# Global vars:
STEP = 2
QUIVER = (0,255,255) # yellow

def draw_flow_hyst(thh, img, flow, step=STEP):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    X1=[]
    Y1=[]
    sig = 0
    lines = []
    for ii in range(len(y)):
        fx, fy = 2*flow[y[ii], x[ii]].T
        if np.abs(fx)>thh or np.abs(fy)>thh:
            X1 = x[ii]
            Y1 = y[ii]
            lines = np.vstack([x[ii], y[ii], x[ii] + fx, y[ii] + fy]).T.reshape(-1, 2, 2)
            sig = ii+1
            break
    for ii in range(sig, len(y)):
        fx, fy = 2*flow[y[ii], x[ii]].T
        if np.abs(fx)>thh or np.abs(fy)>thh:
            X1 = np.append(X1, x[ii])
            Y1 = np.append(Y1, y[ii])
            line = np.vstack([x[ii], y[ii], x[ii] + fx, y[ii] + fy]).T.reshape(-1, 2, 2)
            lines = np.vstack((lines, line))
    
    if len(lines)==0:
        print('lista vacia')
    else:
        lines = np.int32(lines + 0.5)
    return lines

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    hsv[..., 0] = np.rad2deg(np.arctan2(fy, fx)) % 180
    bgr1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)    
    return bgr, bgr1



def mi_gradiente(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    orientation = np.zeros((h, w), np.uint8)
    orientation2 = np.zeros((h, w), np.uint8)
    orientation2 = (np.arctan2(fy, fx) + np.pi)* (180 / np.pi / 2)
    orientation = np.rad2deg(np.arctan2(fy, fx)) % 180
    v = np.sqrt(fx * fx + fy * fy)
    magnitude = np.zeros((h, w), np.uint8)
    magnitude = np.minimum(v * 4, 255)
    
    # fy = 0.1
    # fx =-1
    # print((np.arctan2(fy, fx) + np.pi)* (180 / np.pi / 2))
    # print(np.rad2deg(np.arctan2(fy, fx)) % 180)    
    
    return magnitude, orientation, orientation2




'''
inicializo variable
'''

# Detectron2 pot detection

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch

CONFIG_COCO = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
MODEL_COCO = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Initialize coco predictor
cfg_coco = get_cfg()
cfg_coco.merge_from_file(CONFIG_COCO)
cfg_coco.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_coco.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
cfg_coco.MODEL.WEIGHTS = MODEL_COCO
coco_predictor = DefaultPredictor(cfg_coco) 


thresh = 10 # initial threshold
n_acumula =  4


#percent by which the image is resized
scale_percent = 1

ctte = 1
   

source_window = 'Farneback'
cv2.namedWindow(source_window, cv2.WINDOW_AUTOSIZE)


cuenta = 1

 
lines = []


# # cap = cv2.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/2020/Videos/videoplayback.mp4")
# cap = cv2.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/Videos/videos de campana/DCA6327319E6_2020_07_03_211702.mp4")
# cap = cv2.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/Videos/videos de campana/DCA6327319E6_2020_12_11_141957.mp4")
# cap = cv2.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/Videos/videos de campana/DCA6327319E6_2020_09_24_130541.mp4")
# cap = cv2.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/Videos/videos de campana/DCA6327319E6_2020_12_11_141957.mp4")
# # cap = cv2.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/2021/Imagenes y videos/onion.mp4")


# cap = cv2.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/Videos/videos de campana/DCA6327319BA_2021_01_21_110822.mp4")
# #frame = 7700


cap = cv2.VideoCapture("../../actions/videos/DCA6327319E6_2020_07_03_211702.mp4")
frame = 1300

cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)


if not cap.isOpened():
    print('Unable to open' )
    exit(0)  

ret, image_np = cap.read()

#calculate the 50 percent of original dimensions
width = int(image_np.shape[1] * scale_percent )
height = int(image_np.shape[0] * scale_percent )
# dsize
dsize = (width, height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # set video extension type
codec = cv2.VideoWriter_fourcc(*'DIVX')
save_as = "out_videos"
video_writer1 = None
video_writer2 = None
#cv2.VideoWriter(save_as + '/OF_hog 2021_01_21_110822_4_4.mp4', fourcc, 20.0, (250, 250)) # path_name, video type, frame rate, (image_width, image_height)
#video_writer2 = cv2.VideoWriter(save_as + '/OF_color 2021_01_21_110822_4_4.mp4', codec, 20.0, dsize) # path_name, video type, frame rate, (image_width, image_height)


image_np = cv2.resize(image_np, dsize)
gray1 = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

ret, image_np = cap.read()    
image_np = cv2.resize(image_np, dsize)
gray2= cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

vis = gray2

flowFB =cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
flow = ctte*flowFB



i=0
while(cap.isOpened()):
    
    if i > 100:
        break
    
    i=i+1
    gray1 = gray2
    ret, image_np = cap.read()
    if ret == False:
        break
    
    try:
        image_np = cv2.resize(image_np, dsize)
        gray2 = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    except:
        print("Error converting to GRAY")
        break
 
    
    print('vamos por la imagen: ', i)

        
    flowFB =cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    auxx=draw_flow_hyst(thresh, gray1, flowFB)


    if len(lines)==0:
        print('lista vacia')
        lines = auxx


    if cuenta % n_acumula == 0:
        cuenta = 1     
        cv2.polylines(vis, lines, 0, QUIVER, thickness=1)
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
        cv2.imshow(source_window, vis)


        if not video_writer2:        
            video_writer2 = cv2.VideoWriter(save_as + '/OF_vis_1300 DCA6327319E6_2020_07_03_211702_2_4.mp4', codec, 20.0, dsize) # path_name, video type, frame rate, (image_width, image_height)            video_writer1 = cv2.VideoWriter(save_as + '/OF_hog_1300 DCA6327319E6_2020_07_03_211702_2_4.mp4', codec, 20.0, (width, height))
        video_writer2.write(vis) 
        

        vis = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
        lines = auxx

        flow_HSV,  flow_HSV2 = draw_hsv(flow)


        ######################################################################        
        ######################################################################        
        
        # aqui es donde tenemos que detectar la elipse        
        roi_flow = flow[80:400, 290:610, :]
        boxes = detect_pots(image_np, coco_predictor)

        if len(boxes) > 0:
            roi = boxes[0]
            roi_flow = flow[roi[0]:roi[2], roi[1]:roi[3]]
            cv2.imshow('ROI', gray1[roi[0]:roi[2], roi[1]:roi[3]]) 
        
        ######################################################################        
        ######################################################################        


        modulo, argumento, argumento2 = mi_gradiente(roi_flow)
        #cv2.imshow('flow HSV', flow_HSV)        
        cv2.imshow('flow HSV1', flow_HSV2)       
     
   
        normalized_blocks, hog_image = mi_hog.hog(modulo, argumento2)
      
           
        flow = ctte*flowFB
          

        hog_image = np.uint8(hog_image)
        hog_image = cv2.cvtColor(hog_image, cv2.COLOR_GRAY2RGB)     
        cv2.imshow('hog', hog_image)
        if not video_writer1:        
            height, width, color = hog_image.shape
            video_writer1 = cv2.VideoWriter(save_as + '/OF_hog_1300 DCA6327319E6_2020_07_03_211702_2_4.mp4', codec, 20.0, (width, height))
        video_writer1.write(hog_image) 
        #video_writer2.write(vis) 



    else:
        cuenta = cuenta + 1
        flow += ctte*flowFB
        if len(auxx)==0 or len(lines)==0:            
            print('lista vacia')
        else:
            lines = np.vstack((lines, auxx))



    keyboard= cv2.waitKey(50) & 0xFF

    if  keyboard == ord('s'):
        cv2.imwrite('OF_frame.jpg', vis)            
    elif    keyboard == 'q' or keyboard == 27:
        break

   
cap.release()
video_writer1 and video_writer1.release()
video_writer2 and video_writer2.release()
cv2.destroyAllWindows()
