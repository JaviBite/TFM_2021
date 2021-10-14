
import json
import sys, os
import codecs, unicodedata
import re, random
import cv2
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

# Ours

from cv_scripts.pot_det import detect_pots
from cv_scripts.flow_hog import mi_gradiente
from cv_scripts.libs import mi_hog

import argparse

# Return the region of interest top-left and bottom-right corners
# Call only with the first frame of each fragment, then use cropROI with the returned values.
# Format: (X0, Y0, w, h)
def getROI(frame, coco_pred, padding, width, height, flow):

    # x1,x2,y1,y2 = funcioncarlos(frame)
    pots = detect_pots(frame, coco_pred, padding)

    if len(pots) > 0:

        # Get the pot with the most flow near it
        most_flow_index = 0
        if len(pots) > 1:

            averages = []
            for pot in pots:
                x1, y1, x2, y2 = pot

                c = [x1+int(x2/2), y1+int(y2/2)]
                areax = [slice(c[1]-5,c[1]+5), slice(c[0]-5,c[0]+5), 0]
                areay = [slice(c[1]-5,c[1]+5), slice(c[0]-5,c[0]+5), 1]
                
                fx, fy = flow[tuple(areax)], flow[tuple(areay)] 
                v = np.sqrt(fx * fx + fy * fy)

                averages.append(np.average(v))

            #print(averages)
            most_flow_index = np.argmax(averages)
            #print(most_flow_index)
            
        x1, y1, x2, y2 = pots[most_flow_index]

        if x1 < 0 : x1 = 0
        if x1+x2 >= width : x2 = width - x1
        if y1 < 0 : y1 = 0
        if y1+y2 >= height : y2 = height - y1

        # Padding alredy done on detect_pots()
        #x1,x2,y1,y2 = x1-padding,x2+padding,y1-padding,y2+padding

        return x1, y1, x2, y2

    else:

        return None

def getVidPath(jsondata, localpath, vid):
    namepath = jsondata['file'][vid]['fname']
    return localpath + namepath

def runFrag(frag, args, coco_predictor):

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

        if final_frame - init_frame < args.frames:
            cap.release()
            return None

        action_noum = frag['act']

        action = action_noum.split(" ")[0]

        #The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
        #Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
        #The second argument defines the frame number in range 0.0-1.0
        cap.set(cv2.CAP_PROP_POS_FRAMES,init_frame)
        
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video  file")
            return None
        
        # Read until video is completed
        frame_i = init_frame
        final_frame = frame_i + args.frames

        # Corners for ROI
        roix1, roix2, roiy1, roiy2 = 0,0,0,0

        roi_window = None
        not_roi_count = 0

        CTTE = 1 # Constat for CTTE * flow

        flow_count = 0
        last_gray_frames = [None, None]

        # First frame for flow
        ret, frame = cap.read()
        last_gray_frames[1] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        sequence = []
        while(cap.isOpened() and not_roi_count < 5):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if frame_i <= final_frame and ret == True: 

                # Update Last Frames
                last_gray_frames[0] = last_gray_frames[1]
                last_gray_frames[1] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #Calcular flujo optico
                flowFB = cv2.calcOpticalFlowFarneback(last_gray_frames[0], last_gray_frames[1], 
                                None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                #DONE Detectar la ROI una vez para todo el fragmento
                if roi_window is None:
                    roi = getROI(frame, coco_predictor, args.padding, width, height, flowFB)

                    if roi is not None:

                        x1, y1, x2, y2 = roi
                        roi_window = [slice(y1,y1+y2), slice(x1,x1+x2)]
                    
                    else:

                        not_roi_count = not_roi_count + 1

                else:

                    img_roi = frame[tuple(roi_window)]
                    img_roi = cv2.resize(img_roi,(args.dimension,args.dimension))
                    #img_roi = cv2.resize(img_roi,(args.dimension,args.dimension))

                    #Acomular y hacer histograma
                    if flow_count >= args.flow_acc:
                        flow_count = 0

                        roi_flow = flow[tuple(roi_window)]
                        roi_flow = cv2.resize(roi_flow,(args.dimension,args.dimension))
                        modulo, argumento, argumento2 = mi_gradiente(roi_flow)
                        normalized_blocks = mi_hog.hog(modulo, argumento2, number_of_orientations=9, pixels_per_cell=(16, 16), 
                                                                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False)

                        #Add hog features to sequence
                        sequence.append(normalized_blocks)
                    
                    else:
                        flow_count = flow_count + 1
                        
                        if flow_count == 1:
                            flow = CTTE * flowFB
                        else:
                            flow += CTTE * flowFB

                frame_i =  frame_i + 1
            
            # Break the loop
            else: 
                break

        cv2.destroyAllWindows()
        cap.release()

        return (sequence, frag['class'])

def main():

    #Default arguments
    MAX_WORKERS = cv2.getNumberOfCPUs() - 1

    FLOW_STEP = 2
    FLOW_ACC = 4      # 4+1 frames acc

    ROI_DIM = 250
    DET_PAD = 0

    FRAMES_PER_SEQ = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", type=str, help="Path to the dataset json")
    parser.add_argument('videos_folder', nargs='?', default="none", help="Path to the videos folder (default by dataset json)")
    parser.add_argument("out_file", type=str, default="out", help="Save training data")
    parser.add_argument('-i',"--std_inp", type=str, nargs='?', help="Get the imput from commands instad on the execution")
    parser.add_argument('-r',"--random_order", action="store_true", help="Random order of video framgets to save")
    parser.add_argument('-vis',"--visualize", action="store_true", help="Visualize the HOG and ROI")
    parser.add_argument('-mf',"--max_fragments", type=int, default=None, help="Max number of fragments to save")
    parser.add_argument('-p',"--padding", type=int, default=DET_PAD, help="Padding for the dettection zone")
    parser.add_argument('-dim',"--dimension", type=int, default=ROI_DIM, help="Dimenson in pixels of the output square video")
    parser.add_argument('-f',"--frames", type=int, default=FRAMES_PER_SEQ, help="Frames per sequence")
    parser.add_argument('-fa',"--flow_acc", type=int, default=FLOW_ACC, help="Flow frames accomulation")

    args = parser.parse_args()
    out_file = args.out_file

    max_out_frags = args.max_fragments
    random_order = args.random_order

    VIS = args.visualize
    
    file1 = args.json_dir

    # Opening JSON file
    f1 = open(file1, encoding='utf-8')
    
    # returns JSON object as 
    # a dictionary
    data1 = json.load(f1)
    f1.close()

    if args.videos_folder == "none":
        localpath = data1['config']['file']['loc_prefix']['1']
    else:
        localpath = args.videos_folder
    
    words = args.std_inp
    if words is None:
        print("Palabras a buscar (separadas por coma): ")
        words = input()

    words = [x.strip() for x in words.split(',')]
    print("Classes: ", words)

    # Save metadata
    metafile = out_file.split('.')[0] + "_metadata.txt"
    with open(metafile, "w") as f:
        for idx, class_name in enumerate(words):
            f.write(str(idx) + " : " + class_name + "\n")

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

                to_append = value
                to_append['match'] = words.index(word)
                video_data[vid].append(to_append)
                break

    #videos = dict(sorted(video_ids.items(), reverse=False))
    #for vid, value in zip(videos, videos.values()):
    #    print(vid, value)

    repVideos = dict(sorted(video_data.items(), reverse=False))
    out_name = word.replace(" ","_")
    frag_count = 0

    fragments = []
    for vid, data in zip(repVideos, repVideos.values()):

        videoPath = getVidPath(data1, localpath, str(vid)) 
        for elem in data:
            if len(elem['z']) == 2:
                fragments.append({'time':elem['z'], 'act':elem['av']['1'], 'vpath':videoPath, 'class':elem['match']})

    if random_order:
        random.shuffle(fragments)

    # Initialize coco predictor
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    CONFIG_COCO = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    MODEL_COCO = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    SCORE_THRESH_TEST = 0.2

    cfg_coco = get_cfg()
    cfg_coco.merge_from_file(CONFIG_COCO)
    cfg_coco.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_coco.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST  # set threshold for this model
    cfg_coco.MODEL.WEIGHTS = MODEL_COCO
    coco_predictor = DefaultPredictor(cfg_coco)

    total_fragments = len(fragments) if max_out_frags is None else max_out_frags

    t = tqdm(total=total_fragments)
    X = []
    y = []

    # Create pool of workers
    pool = Pool(processes=MAX_WORKERS) 

    # Run workers
    for i in range(0,len(fragments),MAX_WORKERS):
        
        multiple_results = []
        for j in range(0,MAX_WORKERS)
            if i+j < len(fragments):
                multiple_results.append(pool.apply_async(runFrag, ([fragments[i+j], args, coco_predictor])))

        # End framgnet processing
        for res in multiple_results:
            ret = res.get() 

            if ret is not None:

                sequence, class_id = ret

                #Sequence and action
                y.append(class_id)
                X.append(np.array(sequence))
                
                frag_count = frag_count + 1
                t.update()

        if max_out_frags is not None and frag_count >= max_out_frags:
            break

    # Write File
    #X = X.reshape(total_fragments, 9*16*16*3*3, 1)

    y = np.array(y)
    X = np.array(X)

    np.savez(out_file + ".npz", a=X, b=y)

if __name__ == "__main__":
    main()