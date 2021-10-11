
import json
import sys, os
import codecs, unicodedata
import re, random
import cv2
import numpy as np
from cv_scripts.pot_det import detect_pots
import torch

import argparse

# Return the region of interest top-left and bottom-right corners
# Call only with the first frame of each fragment, then use cropROI with the returned values.
def getROI(frame, coco_pred, padding, width, height):

    # x1,x2,y1,y2 = funcioncarlos(frame)
    pots = detect_pots(frame, coco_pred, padding)
    print(pots)

    if len(pots) > 0:

        x1,x2,y1,y2 = pots[0]

        if x1<0 : x1 = 0
        if x2>=width : x2 = width - 1
        if y1<0 : y1 = 0
        if y2>=height : y2 = height - 1

        # Padding alredy done on detect_pots()
        #x1,x2,y1,y2 = x1-padding,x2+padding,y1-padding,y2+padding

        return x1,x2,y1,y2

    else:

        return None

# Crop the region of interest image from a given frame (it corresponds to the pan or cup zone)
# Call once the ROI top-left and bottom-right corners have been returned by getROI in the first frame.
def cropROI(frame,x1,x2,y1,y2):

    ret = frame[slice(y1,y1+y2), slice(x1,x1+x2)]

    return ret

def getVidPath(jsondata, localpath, vid):
    namepath = jsondata['file'][vid]['fname']
    return localpath + namepath

def main():

    #Default arguments
    FLOW_STEP = 2
    FLOW_ACC = 4

    ROI_DIM = 250
    DET_PAD = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", type=str, help="Path to the dataset json")
    parser.add_argument('videos_folder', nargs='?', default="none", help="Path to the videos folder (default by dataset json)")
    parser.add_argument("out_file", type=str, default="out.csv", help="Save training data")
    parser.add_argument('-r',"--random_order", action="store_true", help="Random order of video framgets to save")
    parser.add_argument('-mf',"--max_fragments", type=int, default=None, help="Max number of fragments to save")
    parser.add_argument('-p',"--padding", type=int, default=DET_PAD, help="Padding for the dettection zone")
    parser.add_argument('-dim',"--dimension", type=int, default=ROI_DIM, help="Dimenson in pixels of the output square video")

    args = parser.parse_args()
    out_file = args.out_file

    max_out_frags = args.max_fragments
    random_order = args.random_order
    
    file1 = args.json_dir

    # Opening JSON file
    f1 = open(file1, encoding='utf-8')
    
    # returns JSON object as 
    # a dictionary
    data1 = json.load(f1)

    if args.videos_folder == "none":
        localpath = data1['config']['file']['loc_prefix']['1']
    else:
        localpath = args.videos_folder

    word = 0
    print("Palabras a buscar (separadas por coma): ")
    words = input()

    words = [x.strip() for x in words.split(',')]
    print("Classes: ", words)

    video_ids = {}
    video_data = {}
    for tag, value in zip(data1['metadata'], data1['metadata'].values()):
        #print(tag, value)

        action = str(value['av']['1'])
        vid = int(value['vid'])

        #if word in action and vid not in video_ids:
        for word in words:
            if re.search(word,action) is not None:
                if vid not in video_ids:
                    video_ids[vid] = []
                
                if action not in video_ids[vid]:
                    video_ids[vid].append(action)

                if vid not in video_data:
                    video_data[vid] = []

                video_data[vid].append(value)
                break


    ''''videos = dict(sorted(video_ids.items(), reverse=False))
    for vid, value in zip(videos, videos.values()):
        print(vid, value)'''

    repVideos = dict(sorted(video_data.items(), reverse=False))
    out_name = word.replace(" ","_")
    frag_count = 0

    fragments = []
    for vid, data in zip(repVideos, repVideos.values()):

        videoPath = getVidPath(data1, localpath, str(vid)) 
        for elem in data:
            if len(elem['z']) == 2:
                fragments.append({'time':elem['z'], 'act':elem['av']['1'], 'vpath':videoPath})

    if random_order:
        random.shuffle(fragments)

    fout = open(out_file, "w")

    # Initialize coco predictor
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    CONFIG_COCO = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    MODEL_COCO = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    SCORE_THRESH_TEST = 0.3

    cfg_coco = get_cfg()
    cfg_coco.merge_from_file(CONFIG_COCO)
    cfg_coco.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_coco.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST  # set threshold for this model
    cfg_coco.MODEL.WEIGHTS = MODEL_COCO
    coco_predictor = DefaultPredictor(cfg_coco)

    for frag in fragments:

        # Create a VideoCapture object and some useful data
        videoPath = frag['vpath']
        cap = cv2.VideoCapture(videoPath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps

        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

        init_frame = int(frag['time'][0] * fps)
        final_frame = int(frag['time'][1] * fps)
        #frame_no = init_frame/frame_count

        action_noum = frag['act']

        action = action_noum.split(" ")[0]

        #The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
        #Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
        #The second argument defines the frame number in range 0.0-1.0
        cap.set(cv2.CAP_PROP_POS_FRAMES,init_frame)
        
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video  file")
            continue
        
        # Read until video is completed
        frame_i = init_frame

        # Corners for ROI
        roix1, roix2, roiy1, roiy2 = 0,0,0,0

        sequencia = [[]]

        got_roi = False
        not_roi_count = 0
        while(cap.isOpened() and not_roi_count < 5):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if frame_i <= final_frame and ret == True: 
                
                #DONE Detectar la ROI una vez para todo el fragmento
                if not got_roi:
                    roi = getROI(frame, coco_predictor, args.padding, width, height)

                if roi is not None:
                    
                    got_roi = True
                    roix1,roix2,roiy1,roiy2 = roi

                    img_roi = cropROI(frame,roix1,roix2,roiy1,roiy2)
                    #img_roi = cv2.resize(img_roi,(args.dimension,args.dimension))

                    # Visualization
                    visualiced = False
                    if True and not visualiced:
                        cv2.imshow("ROI", img_roi)
                        cv2.waitKey(10)
                        visualiced = True

                    #TODO Calcular flujo optico

                    #TODO Acomular y hacer histograma

                    #TODO Guardar histograma en sequencia

                else:

                    not_roi_count = not_roi_count + 1

                frame_i =  frame_i + 1
            
            # Break the loop
            else: 
                break

        # End framgnet processing

        #TODO Escribir secuencia y la classe (action)
        sequencia = "TODO"
        fout.write(sequencia + action + "\n")
        
        cap.release()
        frag_count = frag_count + 1

        if max_out_frags is not None and frag_count >= max_out_frags:
            break

        


    # Closing file
    fout.close()
    f1.close()

if __name__ == "__main__":
    main()