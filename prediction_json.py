#python prediction_json.py BSH_firsthalf_0.2_pots_forml_nogit.json out_datasets/40-4_p20_d250_ml_metadata.json out_datasets/40-4_p20_d250_ml_test.npz ../models/bilstm_ml/out_model_bilstm.h5 -c 1 -o out_pred_videos -off 1 -w 0.7

import json, argparse, os, sys, re, random

import cv2

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

    def clear(self):
        self.seq.clear()        

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

    THRES = 0.4
    
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
    CLASS_NAMES = metadata['classes']
    frag_indx = range(len(all_data))
    if COUNT is not None:
        frag_indx = random.sample(range(len(all_data)), COUNT)

    # Iterate thorugh samples
    t=tqdm(total=len(frag_indx))

    list_yhat_avg = []
    list_yhat_mean = []
    list_yhat_freq = []
    list_yhat_t2 = []
    list_yhat_t3 = []
    list_yhat_t4 = []

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
            out_path = os.path.join(OUT_FOLDER, str(vid)+"_"+str(int(init_frame_o))+".mp4")
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
        frame_i = init_frame_o+1
        predictions = []
        in_action = []
        in_action_frames = False
        while(not DEBUG and cap.isOpened() and frame_i <= (end_frame_o + 25)):
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
                        in_action.append(int(abs(end_frame - frame_i)) < int((LEN_SEQ * FLOW_ACC)/2))
                        add_count = 0



                in_action_frames = frame_i > init_frame and frame_i < end_frame

            if ret:

                # Save video frame if out set
                if vid_out is not None:
                    
                    # Show legend on first frames
                    if len(predictions) <= 1:
                        legend_text = "[STIR, ADD, FLIP, OTHERS]"
                        (w, h), _ = cv2.getTextSize(legend_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        #cv2.putText(frame, legend_text, (int(width-w-20), int(height-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


                    # Add text frame number
                    colorbg = (200,200,200) if not in_action_frames else (0,255,0)
                    colorfont = (0,0,0)
                    (w, h), _ = cv2.getTextSize(str(frame_i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (0, 0), (w + 4, int(h+5)), colorbg, -1)
                    cv2.putText(frame, str(int(frame_i)), (3, h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorfont, 1)

                    # Add text of the last prediction
                    if len(predictions) > 0:
                        pred_str = '[{:s}]'.format(', '.join(['{:.2f}'.format(x) for x in predictions[-1]]))

                        class_names = []
                        for idx, value in enumerate(predictions[-1]):
                            if value > THRES:
                                class_names.append(CLASS_NAMES[idx])

                        text = "Pred: ("+ str(len(predictions)) +"): " + pred_str + " : " + str(class_names)
                        color = (0, 255, 0) if in_action[-1] else (255, 255, 255)
                        cv2.putText(frame, text, (10, int(height-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        text = "          [STIR, ADD, FLIP, OTHERS]"
                        cv2.putText(frame, text, (10, int(height-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        avg = sum(np.array(predictions)/len(predictions))
                        avg_str = '[{:s}]'.format(', '.join(['{:.2f}'.format(x) for x in avg]))
                        text = "Avg: ("+ str(len(predictions)) +"): " + avg_str
                        #cv2.putText(frame, text, (10, int(height-50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        yhat = (avg > THRES).astype(int)
                        yhat_str = '[{:s}]'.format(', '.join([str(int(x)) for x in yhat]))
                        text = "Th: ("+ str(len(predictions)) +"): " + yhat_str
                        #cv2.putText(frame, text, (10, int(height-70)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

        # Average
        avg = sum(np.array(predictions)/len(predictions))

        # Most frequent
        thresholded = np.array(np.array(predictions) > THRES).astype(int)
        
        print(thresholded)
        pairs, counts = np.unique(thresholded, axis=0, return_counts=True)
        most_frequent = pairs[counts.argmax()]

        # Two consecuvites
        two_con = None
        three_con = None
        four_con = None
        value = thresholded[0]
        count_same = 0
        for elem in thresholded[1:]:
            if (elem == value).all():
                count_same += 1
            else:
                count_same = 0
                value = elem
            
            if count_same == 1 and two_con is None:
                two_con = elem
            if count_same == 2 and three_con is None:
                three_con = elem
            if count_same == 3 and four_con is None:
                four_con = elem

        if two_con is None:
            two_con = np.zeros(4)
        if three_con is None:
            three_con = np.zeros(4)
        if four_con is None:
            four_con = np.zeros(4)

        print("Most frequent: ", most_frequent)
        print("Two consecuvites: ", two_con)
        print("Tree consecuvites: ", three_con)
        print("Four consecuvites: ", four_con)
        print("Average: ", (avg > THRES))


        yhat = (avg > THRES).astype(int)
        good = (yhat == label).all()

        list_yhat_avg.append((avg > THRES).astype(int))
        list_yhat_freq.append(most_frequent)
        list_yhat_t2.append(two_con)
        list_yhat_t3.append(three_con)
        list_yhat_t4.append(four_con)

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

        # Graph predictions
        if OUT_FOLDER is not None:
            img_out_path = os.path.join(OUT_FOLDER, str(vid)+"_"+str(int(init_frame_o))+".png")

            fig, ax = plt.subplots()

            predictions = np.matrix(predictions)
            time_frames = np.arange(0, len(predictions), 1)
            ax.plot(time_frames, predictions[:,0], color = 'green', label = 'Stir')
            ax.plot(time_frames, predictions[:,1], color = 'red', label = 'Add')
            ax.plot(time_frames, predictions[:,2], color = 'blue', label = 'Flip')
            ax.plot(time_frames, predictions[:,3], color = 'yellow', label = 'Others')
            ax.set_xlabel("Time stamps")
            ax.set_ylabel("Score")
            ax.legend(loc = 'upper left')
            plt.savefig(img_out_path)
        
        t.update()




    
    t.close()


    names = ["Average","Most Frequent","Two Consecutives","Three Consecutives","Four Consecutives"]
    lists = [list_yhat_avg, list_yhat_freq, list_yhat_t2, list_yhat_t3, list_yhat_t4]
    for idx, list_yhat in enumerate(lists):

        print("==========================================================")
        print("================   " + names[idx] +  "   =====================")
        print("==========================================================")
        acc = accuracy_score(list_y, list_yhat, normalize=True)
        f1_s = f1_score(list_y, list_yhat, average='weighted')
        precision = precision_score(list_y, list_yhat, average='weighted')
        recall = recall_score(list_y, list_yhat, average='weighted')

        # Priny statistics
        print("Accuracy: ", acc)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1_s)


    print("\nTotal Prob: ", total_prob/len(frag_indx))
    print("Total Pred: ", total_pred/len(frag_indx))

if __name__ == '__main__':
    main()