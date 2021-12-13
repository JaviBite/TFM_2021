#python prediction_json.py BSH_firsthalf_0.2_pots_forml_nogit.json out_datasets/40-4_p20_d250_ml_metadata.json out_datasets/40-4_p20_d250_ml_test.npz ../models/bilstm_ml/out_model_bilstm.h5 

import json, argparse, os, sys, re, random

import cv2

from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

import traceback, logging

from keras.utils.all_utils import to_categorical
from keras.models import load_model

# Ours
from cv_scripts.pot_det import detect_pots
from cv_scripts.pot_det_cv import detect_pots_cv, encuentra_box, iou
from cv_scripts.flow_hog import mi_gradiente, draw_hsv
from cv_scripts.libs.mi_hog import normalize
from cv_scripts.libs import mi_hog


def getVidPath(jsondata, localpath, vid):
    namepath = jsondata['file'][str(vid)]['fname']
    return localpath + namepath

def main():

    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, help="Path to the dataset .json")
    parser.add_argument("meta_file", type=str, help="Path to the dataset metadata .json")
    parser.add_argument("data_file", type=str, help="Path to the dataset samples .npz")
    parser.add_argument("model_file", type=str, help="Path to the keras model .h5")
    parser.add_argument('videos_folder', nargs='?', default="none", help="Path to the videos folder (default by dataset json)")
    parser.add_argument('-vis',"--visualize", action="store_true", help="Visualize the HOG and ROI")
    parser.add_argument('-d',"--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()
    
    # Parsed values
    VID_FOLDER = args.videos_folder
    VIS = args.visualize or args.debug
    
    MODEL_FILE = args.model_file
    DATA_FILE = args.data_file
    META_FILE = args.meta_file
    JSON_FILE = args.json_file

    DIM = 250
    DEBUG = args.debug
    FLOW_ACC = 4
    LEN_SEQ = 10
    FRAMES_PER_SEQ = FLOW_ACC * LEN_SEQ

    START_FRAMES = FRAMES_PER_SEQ * 2
    END_FRAMES = FRAMES_PER_SEQ * 2

    # Opening M file
    f1 = open(META_FILE, encoding='utf-8')
    metadata = json.load(f1)
    f1.close()

    f1 = open(JSON_FILE, encoding='utf-8')
    data = json.load(f1)
    f1.close()

    if VID_FOLDER == "none":
        localpath = data['config']['file']['loc_prefix']['1'][8:]
    else:
        localpath = args.videos_folder
    
    class_names = metadata['classes']
    print("Classes: ", class_names)

    N_CLASSES = len(class_names)

    # Get labels
    files = np.load(DATA_FILE, allow_pickle=True)
    _, labels = files['a'], files['b']

    # Get the model
    model = load_model(MODEL_FILE)

    # Iterate thorugh samples
    for idx, value in enumerate(metadata['samples_test']):

        vid = int(value['vid'])
        init_frame = int(value['frames'][0])
        end_frame = int(value['frames'][1])
        roi = value['roi']
        label = labels[idx]

        video_path = getVidPath(data, localpath, vid)

        #Roi
        x1, y1, x2, y2 = roi
        roi_window = [slice(y1,y1+y2), slice(x1,x1+x2)]

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`                                                                                                                                                                                 


        init_frame_o = init_frame - START_FRAMES
        end_frame_o = end_frame + END_FRAMES

        print("Video interval: ", init_frame_o, "-", end_frame_o)

        # Fragments which have the action
        duration_frames = end_frame_o - init_frame_o
        n_fragments = int(duration_frames / FRAMES_PER_SEQ)
        time_action = [0] * n_fragments

        print("Duration: ", duration_frames)

        for i in range(n_fragments):
            current_frame = i * FRAMES_PER_SEQ + init_frame_o
            if current_frame >= init_frame and current_frame <= end_frame:
                time_action[i] = 1

        print("Action timestamps: ", time_action)


        # Set the initial frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_o)
        
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video  file")
            continue

        # Read Initial frame
        ret, frame = cap.read()

        if DEBUG:
            
            img_roi = frame[tuple(roi_window)]
            img_roi = cv2.resize(img_roi,(DIM,DIM))
            cv2.imshow("ROI", img_roi)
            cv2.waitKey(0)
            
        #Initial flow
        last_gray_frames = [None, None]
        img_roi = frame[tuple(roi_window)]
        img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        last_gray_frames[1] = cv2.resize(img_roi,(DIM,DIM))
    
        sequence = []

        # GET FRAMES
        flow_count = 0
        frame_i = init_frame+1
        predictions = []
        while(not DEBUG and cap.isOpened() and frame_i <= end_frame_o):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret: 
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_frame = cv2.resize(gray_frame[tuple(roi_window)],(DIM,DIM))

                # Update Last Frames
                last_gray_frames[0] = last_gray_frames[1]
                last_gray_frames[1] = roi_frame
                
                #Calcular flujo optico
                flowFB = cv2.calcOpticalFlowFarneback(last_gray_frames[0], last_gray_frames[1], 
                                None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                # Visualization
                if VIS:
                    img_roi = frame[tuple(roi_window)]
                    img_roi = cv2.resize(img_roi,(DIM,DIM))
                    cv2.imshow("ROI", img_roi)
                    cv2.imshow("Image", frame)
                    cv2.waitKey(100)


                #Acomular y hacer histograma
                if flow_count == 0:
                    flow = flowFB

                else:
                    flow += flowFB
                
                flow_count = flow_count + 1
                        
                if flow_count >= FLOW_ACC:
                    flow_count = 0

                    modulo, _, argumento2 = mi_gradiente(flow)

                    orientations = 9
                    pixels_per_cell = (16, 16)
                    normalized_blocks = mi_hog.hog(modulo, argumento2, number_of_orientations=orientations, pixels_per_cell=pixels_per_cell, 
                                                            cells_per_block=(3, 3), block_norm='L2-Hys', visualize=VIS)

                    # Visualization
                    if VIS:
                        hog_image = normalized_blocks[1]
                        hog_image = np.uint8(hog_image)
                        hog_image = cv2.cvtColor(hog_image, cv2.COLOR_GRAY2RGB)     
                        cv2.imshow('hog', hog_image)
                        normalized_blocks = normalized_blocks[0]

                    sequence.append(normalize(normalized_blocks))

                    if len(sequence) >= LEN_SEQ:
                        
                        #Predict the sequence
                        n_features = sequence[0].shape[0]
                        x = np.array(sequence).reshape(1, LEN_SEQ, n_features)
                        model_prediction = np.squeeze(model.predict(x))
                        predictions.append(model_prediction)
                        sequence.clear()
            
                frame_i = frame_i + 1

            else:
                # Break the loop
                print("Error not more frames to read!") 
                break
        
        cv2.destroyAllWindows()
        cap.release()

        if VIS:
            print("Action timestamps: ", time_action)
            print("True: ", label)
            print("Predicitons: ", np.array(predictions), "count = ", len(predictions))
            print("Avg Predicitons: ", sum(np.array(predictions)/len(predictions)))

if __name__ == '__main__':
    main()