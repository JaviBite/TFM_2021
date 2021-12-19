#python prediction_json.py BSH_firsthalf_0.2_pots_forml_nogit.json out_datasets/40-4_p20_d250_ml_metadata.json out_datasets/40-4_p20_d250_ml_test.npz ../models/bilstm_ml/out_model_bilstm.h5 -c 1 -o out_pred_videos

import json, argparse, os, sys, re, random

import cv2

from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score

from keras.utils.all_utils import to_categorical
from keras.models import load_model

# Ours
from cv_scripts.pot_det import detect_pots
from cv_scripts.pot_det_cv import detect_pots_cv, encuentra_box, iou
from cv_scripts.flow_hog import mi_gradiente, draw_hsv
from cv_scripts.libs.mi_hog import normalize
from cv_scripts.libs import mi_hog

class Window:

    def __init__(self, win_size):
        self.seq = []
        self.win_size = win_size

    def add(self, x):
        self.seq.append(x)
        if len(self.seq) > self.win_size:
            self.seq.pop(0)

    def get(self):
        return self.seq

    def __len__(self):
        return len(self.seq)



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
    parser.add_argument('-c', "--count", type=int, default=None, help="number of videos to process")
    parser.add_argument('-o', '--out', type=str, default=None, help="Path to the output videos folder (default None)")
    parser.add_argument('-off', '--range_offset', type=float, default=1, help="Number of frames to add to the end and beginning of the video")
    parser.add_argument('-w', '--window', type=float, default=0.0, help="Percentage of overlapping between windows")
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

    OUT_FOLDER = args.out
    COUNT = args.count

    DIM = 250
    DEBUG = args.debug
    FLOW_ACC = 4
    LEN_SEQ = 10
    FRAMES_PER_SEQ = FLOW_ACC * LEN_SEQ

    START_FRAMES = FRAMES_PER_SEQ * args.range_offset
    END_FRAMES = FRAMES_PER_SEQ * args.range_offset

    PREDICT_STEPS = int(LEN_SEQ * (1-args.window))

    # Create out folder if not exists
    if OUT_FOLDER is not None:
        if not os.path.exists(OUT_FOLDER):
            os.makedirs(OUT_FOLDER)

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

    # Get the data
    all_data = metadata['samples_test']
    frag_indx = range(len(all_data))
    if COUNT is not None:
        frag_indx = random.sample(range(len(all_data)), COUNT)

    # Iterate thorugh samples
    t=tqdm(len(frag_indx))
    list_yhat = []
    list_y = []
    good_count = 0
    total_pred = [0] * N_CLASSES
    total_prob = [0] * N_CLASSES
    for i in frag_indx:

        value = all_data[i]
        vid = int(value['vid'])
        init_frame = int(value['frames'][0])
        end_frame = int(value['frames'][1])
        roi = value['roi']
        label = labels[i]

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

        # If out folder, create writter
        vid_out = None
        if OUT_FOLDER is not None:
            out_path = os.path.join(OUT_FOLDER, str(vid)+"_"+str(init_frame_o)+".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_out = cv2.VideoWriter(out_path, fourcc, fps, (int(width), int(height)))


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
        window = Window(LEN_SEQ)
        add_count = 0

        # GET FRAMES
        flow_count = 0
        frame_i = init_frame+1
        predictions = []
        while(not DEBUG and cap.isOpened() and frame_i <= (end_frame_o + 40)):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret and frame_i <= end_frame_o: 
                
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

                    nblocks = normalize(normalized_blocks)
                    #sequence.append(nblocks)
                    window.add(nblocks)
                    add_count += 1

                    if len(window) >= LEN_SEQ and add_count >= PREDICT_STEPS:
                        
                        #Predict the sequence
                        sequence = window.get()
                        n_features = sequence[0].shape[0]
                        x = np.array(sequence).reshape(1, LEN_SEQ, n_features)
                        model_prediction = np.squeeze(model.predict(x))
                        predictions.append(model_prediction)
                        add_count = 0

            if ret:

                # Save video frame if out set
                if vid_out is not None:

                    # Add text frame number
                    (w, h), _ = cv2.getTextSize(str(frame_i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (0, 0), (w + 4, int(h+5)), (255,255,255), -1)
                    cv2.putText(frame, str(frame_i), (3, h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                    # Add text of the last prediction
                    if len(predictions) > 0:
                        pred_str = '[{:s}]'.format(', '.join(['{:.2f}'.format(x) for x in predictions[-1]]))
                        text = "Pred ("+ str(len(predictions)) +"): " + pred_str
                        cv2.putText(frame, text, (10, int(height-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    vid_out.write(frame)

                    if VIS:
                        cv2.imshow("Image", frame)
            
            else:
                
                # Break the loop
                print("Error not more frames to read!") 
                break

            frame_i = frame_i + 1
        
        cv2.destroyAllWindows()
        cap.release()


        avg = sum(np.array(predictions)/len(predictions))
        yhat = (avg > 0.4).astype(int)
        good = (yhat == label).all()

        list_yhat.append(yhat)
        list_y.append(label)

        if good:
            good_count = good_count + 1

        if VIS:
            print("\nTrue: ", label)
            print("Predicitons: ", np.array(predictions), "count = ", len(predictions))
            print("Avg Predicitons: ", avg)
            print("Prediction: ", yhat)
            if good:
                print("Correct!!")
            else:
                print("Bad :(")

        total_prob += avg
        total_pred += yhat
        
        t.update()


    
    t.close()

    f1_s = f1_score(list_y, list_yhat, average='weighted')
    precision = precision_score(list_y, list_yhat, average='weighted')
    recall = recall_score(list_y, list_yhat, average='weighted')

    # Priny statistics
    print("Accuracy: ", good_count/len(frag_indx))
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_s)


    print("\nTotal Prob: ", total_prob/len(frag_indx))
    print("Total Pred: ", total_pred/len(frag_indx))



if __name__ == '__main__':
    main()