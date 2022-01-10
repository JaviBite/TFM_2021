import json, argparse, os, sys, re, random

import cv2

from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm


from keras.utils.all_utils import to_categorical
from keras.models import load_model

import warnings

from yacs.config import _YAML_EXTS
warnings.filterwarnings("ignore")

# Initialize coco predictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torch.cuda import is_available

CONFIG_COCO = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
MODEL_COCO = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
SCORE_THRESH_TEST = 0.2

# Ours
from cv_scripts.pot_det import detect_pots
from cv_scripts.pot_det_cv import detect_pots_cv, encuentra_box, iou
from cv_scripts.flow_hog import mi_gradiente, draw_hsv
from cv_scripts.libs.mi_hog import normalize
from cv_scripts.libs import mi_hog

from json_export_flow import closest_rois
from json_export_flow import getMostFlowRoi
from prediction_json import Window


def main():

    NET_MODEL = "../models/bilstm_ml/out_model_bilstm.h5"
    VID = 34#None
    JSON_FILE =  "./BSH_firsthalf_0.2_pots_forml_nogit.json"
    CLASSES = ["remover ","poner (?!olla|sarten|cazo)","voltear ","^(?!remover|interaccion|poner|voltear) "]
    CLASS_NAMES = ["STIR", "ADD", "FLIP", "OTHERS"]
    OUT_FOLDER = "./out_all_videos_pred"

    # Create dir if not exists
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    PREDICT_STEPS = 4
    WINDOW_OVERLAPPING = 0.7
    WRITE_VIDEO = True
    AFTER_TIME = 3.0  #seconds

    VIS = True
    DEBUG = False

    VIS_HOG = False

    END_OFF = 1
    INIT_OFF = 1

    # DO NOT CHANGE
    FPS = 25
    DIM = (250, 250)
    FLOW_ACC = 4
    PPC = (16,16)
    ORIENT = 9
    LEN_SEQ = 10

    # Calculated constas
    PREDICT_STEPS = int(LEN_SEQ * (1 - WINDOW_OVERLAPPING))

    # Load json file
    json_data = json.load(open(JSON_FILE))

    # Iterate through videos
    break_at_end = False
    for vid in range(1,100):

        if VID is None:
            break_at_end = False
            VID = vid

        # Video path
        VIDEO_PATH = None
        localpath = json_data['config']['file']['loc_prefix']['1'][8:]
        for value in json_data['file'].values():

            if int(value['fid']) == VID:
                VIDEO_PATH = localpath + value['fname']
                break
        
        if VIDEO_PATH is None:
            print("Video not found, VID = ", VID)
            return -1


        # Load model
        model = load_model(NET_MODEL)

        # Coco predictor
        '''
        cfg_coco = get_cfg()
        cfg_coco.merge_from_file(CONFIG_COCO)
        cfg_coco.MODEL.DEVICE = 'cuda' if is_available() else 'cpu'
        cfg_coco.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST  # set threshold for this model
        cfg_coco.MODEL.WEIGHTS = MODEL_COCO
        coco_predictor = DefaultPredictor(cfg_coco)'''

        # Open video
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("Error opening video: ", VIDEO_PATH)
            exit()

        # Get video info
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        time_per_frame = 1 / video_fps
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`   
        
        # Video Writer
        if WRITE_VIDEO:
            out_path = f"{OUT_FOLDER}/out_{VID}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vid_out = cv2.VideoWriter(out_path, fourcc, FPS, (int(width), int(height)))

        # Actions stack
        actions = []
        regions = []
        for value in json_data['metadata'].values():
            vid = int(value['vid'])

            if vid == VID:

                # Temporal segment
                if len(value['xy']) == 0:
                    action = str(value['av']['1'])

                    #if word in action and vid not in video_ids:
                    for word in CLASSES:
                        if re.search(word,action) is not None:

                            to_append = value
                            to_append['match'] = CLASSES.index(word)
                            actions.append(to_append)
                            break

                # Region data
                else:

                    regions.append(value)

        # Short regions and actions in chronological order
        regions = sorted(regions, key=lambda d: d['z'][0], reverse=False)
        actions = sorted(actions, key=lambda d: d['z'][0], reverse=False)

        # Skip videos without label data
        if len(actions) == 0 or len(regions) == 0:
            cap.release()
            continue

        # Add regions to actions
        for i in range(len(actions)):

            init_time = float(actions[i]['z'][0])
            possible_rois = closest_rois(regions, init_time)
            actions[i]['rois'] = possible_rois

        #Initial values
        frame_i = 0
        time = 0
        end = False
        window = Window(LEN_SEQ)
        win_add_count = 0

        next_action = actions.pop(0) # Get first action
        actual_roi = None
        flow_count = 0
        flow_acc = np.zeros((DIM[0],DIM[1],2))
        last_gray_frames = [0,0]
        inside_action = False   
        after_action = 0.0

        predictions = []
        pred_per_sec = []
        pred_per_sec_y = []
        sec_timer = 0

        t=tqdm(total=video_length)
        while (cap.isOpened() and not end):

            ret, frame = cap.read()
            if not ret:
                print("No more frames to read")
                break
            
            # Inside the action
            if next_action['z'][0] - INIT_OFF <= time:

                inside_action = True

                if actual_roi is None:

                    # Get the most flow roi
                    # frame_i-1 because we want to get the flow from the previous frame
                    roi = getMostFlowRoi(next_action['rois'], cap, frame_i-1, int(frame_i+20),padding=20, visualize=VIS)
                    x1, y1, x2, y2 = roi
                    pt1 = (x1,y1)
                    pt2 = (x1+x2,y1+y2)
                    actual_roi = [slice(y1,y1+y2), slice(x1,x1+x2)]

                    # Visualization
                    if VIS:
                        # Draw Selected ROI
                        cv2.rectangle(frame, pt1, pt2, (0,255,0), 4)
                        cv2.imshow("ROI", frame)
                        cv2.waitKey(100)

                        if DEBUG:
                            cv2.waitKey(0)
                            # Reset action variables
                            actual_roi = None
                            flow_count = 0
                            flow_acc = np.zeros((DIM[0],DIM[1],2)) 
                            inside_action = False    

                            if len(actions) > 0:
                                next_action = actions.pop(0) 
                            else:                
                                # End of the video
                                end = True
                            continue

                    # Get the initial flow
                    _, frame = cap.read()
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_frame = cv2.resize(gray_frame[tuple(actual_roi)],DIM)
                    last_gray_frames[1] = roi_frame

                    # Next frame for flow
                    _, frame = cap.read()
                
                # Calculate flow
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_frame = cv2.resize(gray_frame[tuple(actual_roi)],DIM)

                # Update Last Frames
                last_gray_frames[0] = last_gray_frames[1]
                last_gray_frames[1] = roi_frame
            
                #Calculate optical flow
                flowFB = cv2.calcOpticalFlowFarneback(last_gray_frames[0], last_gray_frames[1], 
                                None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                # Accumulate flow
                if flow_count < FLOW_ACC:
                    flow_acc += flowFB
                    flow_count += 1

                else:

                    # Calculate HOG
                    modulo, argumento, argumento2 = mi_gradiente(flow_acc)
                    normalized_blocks = mi_hog.hog(modulo, argumento2, number_of_orientations=ORIENT, pixels_per_cell=PPC, 
                                                            cells_per_block=(3, 3), block_norm='L2-Hys', visualize=VIS_HOG)
                    
                    # Visualization
                    if VIS_HOG:
                        hog_image = normalized_blocks[1]
                        hog_image = np.uint8(hog_image)
                        hog_image = cv2.cvtColor(hog_image, cv2.COLOR_GRAY2RGB)     
                        cv2.imshow('hog', hog_image)
                        cv2.waitKey(100)
                        normalized_blocks = normalized_blocks[0]
                    
                    # Normalize and add it to the window
                    nblocks = normalize(normalized_blocks)
                    window.add(nblocks)
                    win_add_count += 1

                    # Reset variables
                    flow_acc = np.zeros((DIM[0],DIM[1],2))
                    flow_count = 0

                #Predict
                if len(window) >= LEN_SEQ and win_add_count >= PREDICT_STEPS:
                            
                    #Predict the sequence
                    sequence = window.get()
                    n_features = sequence[0].shape[0]
                    x = np.array(sequence).reshape(1, LEN_SEQ, n_features)
                    model_prediction = np.squeeze(model.predict(x))
                    predictions.append(model_prediction)
                    win_add_count = 0         

            # End of the action
            if next_action['z'][1] + END_OFF <= time:

                # Reset action variables
                actual_roi = None
                flow_count = 0
                flow_acc = np.zeros((DIM[0],DIM[1],2)) 
                inside_action = False 

                after_action = AFTER_TIME   

                if len(actions) > 0:
                    next_action = actions.pop(0) 
                else:                
                    # End of the video
                    end = True

            sec_timer += time_per_frame

            # Update prediction
            if sec_timer > 1.0:
                sec_timer = 0.0

                pred = [0.0, 0.0, 0.0, 0.0]
                y = pred
                if inside_action or after_action > 0.0:
                    # Mean of last predictions
                    y = to_categorical(next_action['match'], num_classes=len(CLASS_NAMES))
                    if len(predictions) > 5:
                        pred = sum(np.array(predictions[-5:])/5)
                    elif len(predictions) > 0:
                        pred = sum(np.array(predictions[-len(predictions):])/len(predictions))
                
                pred_per_sec.append(pred)
                pred_per_sec_y.append(y)

                if not inside_action and len(window) > 0:
                    # Clean predictions
                    predictions.clear()
                    window.clear()
                    win_add_count = 0

                if not inside_action and after_action > 0.0:
                    after_action -= 1.0

            # Write video
            if WRITE_VIDEO:

                # Add text frame number
                colorbg = (200,200,200) if not inside_action else (0,255,0)
                colorfont = (0,0,0)
                (w, h), _ = cv2.getTextSize(str(frame_i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (0, 0), (w + 4, int(h+5)), colorbg, -1)
                cv2.putText(frame, str(int(frame_i)), (3, h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorfont, 1)

                # Add text of the last prediction
                if len(pred_per_sec) > 0:
                    pred_str = '[{:s}]'.format(', '.join(['{:.2f}'.format(x) for x in pred_per_sec[-1]]))

                    class_names = []
                    for idx, value in enumerate(pred_per_sec[-1]):
                        if value > 0.4:
                            class_names.append(CLASS_NAMES[idx])

                    text = pred_str + " : " + str(class_names)
                    color = (0, 255, 0) if inside_action else (255, 255, 255)
                    cv2.putText(frame, text, (10, int(h+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw ROI
                #if inside_action:
                    # cv2.rectangle(frame, pt1, pt2, (0,255,0), 4)


                vid_out.write(frame)

            # Update time and frames
            frame_i += 1
            time += time_per_frame
            t.update(1)

        # Write predicitons log
        np.savetxt(f"{OUT_FOLDER}/out_{VID}_yhat.txt", np.array(pred_per_sec), fmt='%.2f')
        np.savetxt(f"{OUT_FOLDER}/out_{VID}_y.txt", np.array(pred_per_sec_y), fmt='%.2f')

        cap.release()
        cv2.destroyAllWindows()

        if break_at_end:
            break


    return 0


if __name__ == '__main__':
    main()