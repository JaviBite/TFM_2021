# Script para detectar las sartenes/ollas/etc
# Detectron2 (y tracking despues)

import sys, os
import codecs, unicodedata
import cv2, math
import numpy as np

import argparse
import time
import tqdm

import torch

# Detectron imports
from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes, BoxMode, Instances, pairwise_ioa

# Coco models

CONFIG_COCO = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
MODEL_COCO = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

REFRESH_RATE = 3
PAD = 20
INIT_FRAMES = 30

#CAT_2_TO_1 = {0: 81, 1: 82, 2: 73, 3: 83, 4: 84}

SCORE_THRESH_TEST = 0.75

def detBox_to_Box(box, pad=0):
    box_numpy = np.squeeze(box.tensor.to(torch.device("cpu")).numpy())
    box_numpy = [int(box_numpy[0]-pad), int(box_numpy[1]-pad), 
                 int(box_numpy[2]-box_numpy[0]+pad), int(box_numpy[3]-box_numpy[1]+pad)]
    
    return box_numpy

def draw_pot_elipse(frame, boxes):
    ret = frame

    for idx, box in enumerate(boxes):

        x_lenght = boxes[idx][2]/2
        y_lenght = boxes[idx][3]/2
        
        centre = (int(boxes[idx][0]+x_lenght),int(boxes[idx][1]+y_lenght))
        axesLength = (int(x_lenght), int(y_lenght))

        ret = cv2.ellipse(ret, centre, axesLength, 0, 0, 360, (0,0,255), 3)
        #ret = cv2.rectangle(ret, (box_np[0], box_np[1]), (box_np[0]+box_np[2], box_np[1]+box_np[3]), (0,255,0), 3)

    return ret

def trackeable(class_id, box=None):
    classok = class_id == 41 or class_id == 45

    boxok = True
    areaok = True

    if box is not None:
        boxnp = detBox_to_Box(box)

        areaok = box.area() > (200 * 200)
        boxnp = (boxnp[2] - boxnp[3]) < 100     

    return classok and boxok and areaok

def detect_pots(frame, coco_predictor, pad=0):

    list_of_pots = []
    outputs = coco_predictor(frame)

    for i in range(len(outputs["instances"].pred_classes)):
        box = outputs["instances"].pred_boxes[i][0]
        if trackeable(outputs["instances"].pred_classes[i], box):
            list_of_pots.append(detBox_to_Box(box,pad))

    return list_of_pots

def main():

    REFRESH_RATE = 3
    PAD = 20
    INIT_FRAMES = 30

    SCORE_THRESH_TEST = 0.75
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("in_video", type=str, help="Path to the video that will be processed")
    parser.add_argument("out_video", type=str, nargs="?", default="out.avi", help="Path to save the processed video")

    parser.add_argument("-t", "--threshold", type=float, default=SCORE_THRESH_TEST,
                   help="Threshold for detection")
    
    parser.add_argument("-r", "--refresh_rate", type=int, default=REFRESH_RATE,
                    help="Refresh rate")
                    
    parser.add_argument("-i", "--init_frames", type=int, default=INIT_FRAMES,
                    help="Initialization frames")   

    parser.add_argument("-p", "--padding", type=int, default=PAD,
                    help="Padding for new bounding boxes for tracking initialization") 

    args = parser.parse_args()

    INPUT = args.in_video
    OUTPUT = args.out_video

    REFRESH_RATE = args.refresh_rate
    PAD = args.padding
    INIT_FRAMES = args.init_frames
    SCORE_THRESH_TEST = args.threshold
    
    
    # Extract video properties
    video = cv2.VideoCapture(INPUT)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    MAX_DIST = math.sqrt(width*width + height*height)
    
    print("Frames per second: ",frames_per_second)
    REFRESH_RATE = int(REFRESH_RATE * frames_per_second)
    print("Refresh rate: ",REFRESH_RATE)

    # Initialize video writer
    video_writer = cv2.VideoWriter(OUTPUT, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

    # Initialize coco predictor
    cfg_coco = get_cfg()
    cfg_coco.merge_from_file(CONFIG_COCO)
    cfg_coco.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_coco.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST  # set threshold for this model
    cfg_coco.MODEL.WEIGHTS = MODEL_COCO
    coco_predictor = DefaultPredictor(cfg_coco)

    # Initialize visualizer
    #register_coco_instances("full_test", {}, "./full_test.json", "./dataset")
    #food_dicts = DatasetCatalog.get("full_test")
    
    #metaData = MetadataCatalog.get("full_test")
    #v = VideoVisualizer(metadata=metaData, instance_mode=ColorMode.IMAGE)    
    
    def runOnVideo(video, maxFrames=None):
        """ Runs the predictor on every frame in the video (unless maxFrames is given),
        and returns the frame with the predictions drawn.
        """

        new_id = 0
        readFrames = 0
        updateFrames = INIT_FRAMES
        
        # Deleted trackers counters                    
        use = 0
        use2 = 0
        over = 0
        remScore = 0
        
        # Times
        trackTime = 0
        detTime = 0
        CheckTime = 0
        
        CheckDupTime = 0
        initTime = 0
        
        boxes = None
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            
            vis = frame
            
            #Set refresh frames
            if readFrames % REFRESH_RATE == 0:
                updateFrames = updateFrames + 5                
            
            # Check boxes and init trackers
            if updateFrames > 0:
            
                # Get prediction results for this frame
                tstart = time.time()
                boxes = detect_pots(frame, coco_predictor)
                tend = time.time()
                detTime = detTime + (tend - tstart)


                
            
            # Make sure the frame is colored
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            # Draw a visualization of the predictions using the video visualizer
            #visualization = v.draw_instance_predictions(vis, outputs["instances"].to("cpu"))

            # Convert Matplotlib RGB format to OpenCV BGR format
            #vis = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            if boxes is not None:
                vis = draw_pot_elipse(vis, boxes)

            #cv2.imshow('frame',vis)
            #cv2.waitKey(0)

            #yield visualization
            yield vis

            updateFrames -= 1
            if updateFrames < 0:
                updateFrames = 0
                
            readFrames += 1
            if maxFrames is not None and readFrames > maxFrames:
                break
           
        print("==================")
        """ print("Removed trackers:")
        print("\tUse: ", use, "\n\tUse2: ", use2, "\n\tOver: ", over)
        print("\tScore: ", remScore)"""
        
        trackTime = trackTime / num_frames
        detTime = detTime / num_frames
        CheckTime = CheckTime / num_frames
        
        CheckDupTime = CheckDupTime / (num_frames/REFRESH_RATE)
        initTime = initTime / (num_frames/REFRESH_RATE)
        
        print("Time usage average:")
        print("\tTracking: ", trackTime, "\n\tDetection: ", detTime,
              "\n\tChecking: ", CheckTime, "\n\tCheckingDups: ", CheckDupTime,
              "\n\tMunkers Trackers: ", initTime)

    # Enumerate the frames of the video
    for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames, position=0, leave=True):

        # Write test image
        #cv2.imwrite('POSE detectron2.png', visualization)

        # Write to video file
        video_writer.write(visualization)

    # Release resources
    video.release()
    video_writer.release()
    
    
if __name__ == '__main__':
    main()

