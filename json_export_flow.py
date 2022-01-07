
import json
from scipy import sparse
import re, random
import cv2
import numpy as np
import torch
from tqdm import tqdm

import traceback
import logging

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# Ours

from cv_scripts.pot_det import detect_pots
from cv_scripts.pot_det_cv import detect_pots_cv, encuentra_box, iou
from cv_scripts.flow_hog import mi_gradiente, draw_hsv
from cv_scripts.libs import mi_hog

import argparse

def flow_2_frames(cap, init_frame, look_frame):

    cap.set(cv2.CAP_PROP_POS_FRAMES, look_frame)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flowFB = cv2.calcOpticalFlowFarneback(frame1, frame2, 
                None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

    return flowFB

def detect_pots_det_n(cap, init_frame, n_frames, coco_pred):

    total_pots = []

    for i in range(n_frames):
        ret, frame = cap.read()

        if not ret:
            break

        pots = detect_pots(frame, coco_pred, 0)
            
        for pot in pots:
            add = True
            b1 = pot[0], pot[1], pot[0]+pot[2], pot[1]+pot[3]
            for pot_b in total_pots:

                b2 = pot_b[0], pot_b[1], pot_b[0]+pot_b[2], pot_b[1]+pot_b[3]
                eiou = iou(b1,b2)
                #print(pot_b, pot, eiou)
                if eiou >= 0.5:
                    add = False
                    break
            
            if add:
                total_pots.append(pot)

    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame) 

    ret_pots = []
    for pot in total_pots:
        x1, y1, x2, y2 = pot
        c = (x1+int(x2/2), y1+int(y2/2))
        e = (c,(x2,y2),0)
        ret_pots.append(e)

    return ret_pots

# Return the region of interest top-left and bottom-right corners
# Call only with the first frame of each fragment, then use cropROI with the returned values.
# Format: (X0, Y0, w, h)
def getROI(frame, coco_pred, padding, width, height, flow, VIS):
    
    pots = detect_pots(frame, coco_pred, padding)

    for pot in pots:
        x1, y1, x2, y2 = pot
        c = (x1+int(x2/2), y1+int(y2/2))
        e = (c,(x2,y2),0)
        cv2.ellipse(frame, e, (0,0,255), 4)
    
    if VIS:
        cv2.imshow("LAS ELIPSES FINALES", frame)

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

# Return the region of interest top-left and bottom-right corners
# Call only with the first frame of each fragment, then use cropROI with the returned values.
# Format: (X0, Y0, w, h)
def getROI2(cap, init_frame, n_search_frames, padding, width, height, flow, VIS):

    blurri = 5

    try:
        pots = detect_pots_cv(cap, init_frame, n_search_frames, blurri, VIS)
    except OverflowError as of:
        print("After the Overflow error", of, "skipping")
        pots = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame) 

    if len(pots) > 0:

        # Get the pot with the most flow near it
        most_flow_index = 0
        if len(pots) > 1:

            averages = []
            for pot in pots:
                c, axes, _ = pot

                area = int((axes[0]+axes[1]) / 3)

                areax = [slice(int(c[1]-area),int(c[1]+area)), 
                         slice(int(c[0]-area),int(c[0]+area)), 0]
                areay = [slice(int(c[1]-area),int(c[1]+area)), 
                         slice(int(c[0]-area),int(c[0]+area)), 1]
                
                fx, fy = flow[tuple(areax)], flow[tuple(areay)] 
                v = np.sqrt(fx * fx + fy * fy)

                averages.append(np.average(v[np.nonzero(v)]))

            #print(averages)
            try:
                most_flow_index = np.nanargmax(averages)
            except ValueError as e:
                print("Value error")
                return None
            #print(most_flow_index)
            
        bbox, _ = encuentra_box(pots[most_flow_index])
        x1, y1, x2, y2 = bbox

        x2 = x2-x1
        y2 = y2-y1

        #Padding
        x1,x2,y1,y2 = x1-padding, x2+padding, y1-padding, y2+padding

        if x1 < 0 : x1 = 0
        if x1+x2 >= width : x2 = width - x1
        if y1 < 0 : y1 = 0
        if y1+y2 >= height : y2 = height - y1        

        return int(x1), int(y1), int(x2), int(y2)

    else:

        return None

def getMostFlowRoi(possible_rois, cap, init_frame, search_frame, padding=0, visualize=False):

    cap.set(cv2.CAP_PROP_POS_FRAMES, search_frame)

    ret, frame = cap.read()
    ret, frame2 = cap.read()

    height, width = frame.shape[:2]

    frameflow1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameflow2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    #Get flow
    flow = cv2.calcOpticalFlowFarneback(frameflow1, frameflow2, 
                                        None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # Get the pot with the most flow near it
    most_flow_index = 0
    if len(possible_rois) > 1:

        averages = []
        for pot in possible_rois:

            cx, cy, ax, ay = pot
            c = (cx,cy)
            axes = (ax, ay)

            area = int((axes[0]+axes[1]) / 4)

            areax = [slice(int(c[1]-area),int(c[1]+area)), 
                        slice(int(c[0]-area),int(c[0]+area)), 0]
            areay = [slice(int(c[1]-area),int(c[1]+area)), 
                        slice(int(c[0]-area),int(c[0]+area)), 1]
            
            fx, fy = flow[tuple(areax)], flow[tuple(areay)] 
            v = np.sqrt(fx * fx + fy * fy)

            averages.append(np.nanmean(v[np.nonzero(v)]))

        #print(averages)
        try:
            most_flow_index = np.argmax(averages)
        except ValueError as e:
            print("Value error")
            return None
        #print(most_flow_index)
        
    chosen_pot = possible_rois[most_flow_index]

    elip = ((chosen_pot[0], chosen_pot[1]), (chosen_pot[2]*2, chosen_pot[3]*2), 0)
    bbox, _ = encuentra_box(elip)
    x1, y1, x2, y2 = bbox

    #Padding
    x1,x2,y1,y2 = x1-padding, x2+padding, y1-padding, y2+padding

    x2 = x2-x1  # x2 now is lenght in x axis
    y2 = y2-y1  # y2 now is lenght in y axis

    if x1 < 0 : x1 = 0
    if x1+x2 >= width : x2 = width - x1
    if y1 < 0 : y1 = 0
    if y1+y2 >= height : y2 = height - y1    

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)    

    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

    if visualize:
        for pot in possible_rois:
            c = (pot[0], pot[1])
            axes = (pot[2]*2, pot[3]*2)

            opencv_elipse = (c, axes, 0)
            cv2.ellipse(frame, opencv_elipse, (0,0,255), 4)

        # Draw Selected ROI
        pt1 = (x1,y1)
        pt2 = (x1+x2,y1+y2)
        cv2.rectangle(frame, pt1, pt2, (0,255,0), 4)

        cv2.imshow("LAS ELIPSES FINALES", frame)
    
    return x1, y1, x2, y2

# rois list must be ordered from soon to late        
def closest_rois(rois, time):

    ret_rois = []
    # Get the time of the closest (temporaly) roi
    last_time = rois[0]['z'][0]
    ret_rois.append(rois[0]['xy'][1:])

    for idx in range(1,len(rois)):
        roi_time = rois[idx]['z'][0]

        # If the time is greater than the last time, we have found the closest roi
        if roi_time > time:
            break

        # Same region timestamp
        elif abs(roi_time - last_time) < 0.5:
            ret_rois.append(rois[idx]['xy'][1:])

        # Ne different timestamp
        else:
            ret_rois.clear()

            last_time = rois[idx]['z'][0]
            ret_rois.append(rois[idx]['xy'][1:])

    return ret_rois

def getVidPath(jsondata, localpath, vid):
    namepath = jsondata['file'][vid]['fname']
    return localpath + namepath

def main():

    #Default arguments
    FLOW_STEP = 2
    FLOW_ACC = 4      # 4+1 frames acc

    ROI_DIM = 250
    DET_PAD = 0

    FPS = 25

    FRAMES_PER_SEQ = 50

    RANDOM_STATE = 42

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
    parser.add_argument('-acc',"--flow_accomulate", type=int, default=FLOW_ACC, help="Flows to acommulate before HOG processing")
    parser.add_argument('-ms',"--max_segments", type=int, default=999, help="Max segments per fragment")
    parser.add_argument('-aug',"--augmentation", action="store_true", help="Add data augmentation flipping the frames")
    parser.add_argument('-b',"--balance", action="store_true", help="Force balance between all the classes")
    parser.add_argument('-w',"--window", type=float, default=1.0, help="Percetnage of new data for the new window of fragment.")
    parser.add_argument('-jf',"--just_flow", action="store_true", help="Store flow instead of HOG")
    parser.add_argument('-s',"--split", type=float, default=0.2, help="Split percentage for the dataset in Train and Test")
    parser.add_argument('-d',"--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()
    out_file = args.out_file

    max_out_frags = args.max_fragments
    random_order = args.random_order

    VIS = args.visualize or args.debug
    debug = args.debug

    FLOW_ACC = args.flow_accomulate
    DATA_AUGMENTATION = args.augmentation

    DO_JUST_FLOW = args.just_flow

    MAX_SEGMENTS = args.max_segments
    
    file1 = args.json_dir

    # Opening JSON file
    f1 = open(file1, encoding='utf-8')
    
    # returns JSON object as 
    # a dictionary
    data1 = json.load(f1)
    f1.close()

    if args.videos_folder == "none":
        localpath = data1['config']['file']['loc_prefix']['1'][8:]
    else:
        localpath = args.videos_folder
    
    words = args.std_inp
    if words is None:
        print("Palabras a buscar (separadas por coma): ")
        words = input()

    words = [x.strip() for x in words.split(',')]
    print("Classes: ", words)

    # Save metadata
    metafile = out_file.split('.')[0] + "_metadata.json"

    metadata = {}

    metadata['classes'] = words

    metadata['config'] = {'padding': args.padding,
                          'fragments': -1,
                          'seg_frames': args.frames,
                          'accomulation': FLOW_ACC,
                          'random': random_order,
                          'roi_dim': args.dimension,
                          'augmentation': DATA_AUGMENTATION,}

    video_data = {}
    regions = {}
    for value in data1['metadata'].values():
        #print(tag, value)

        vid = int(value['vid'])

        # Temporal segment
        if len(value['xy']) == 0:
            action = str(value['av']['1'])

            #if word in action and vid not in video_ids:
            for word in words:
                if re.search(word,action) is not None:

                    if vid not in video_data:
                        video_data[vid] = []

                    to_append = value
                    to_append['match'] = words.index(word)
                    video_data[vid].append(to_append)
                    break

        # Region data
        else:

            if vid not in regions:
                regions[vid] = []

            #TODO check good region?
            regions[vid].append(value)

    # Short regions in chronological order
    for vid in regions.keys():
        regions[vid] = sorted(regions[vid], key=lambda d: d['z'][0], reverse=False)

    repVideos = dict(sorted(video_data.items(), reverse=False))
    out_name = word.replace(" ","_")
    frag_count = 0

    fragments = []
    for vid, data in zip(repVideos, repVideos.values()):

        videoPath = getVidPath(data1, localpath, str(vid)) 
        for elem in data:
            if len(elem['z']) == 2:

                # Temporal segment if the fragment duration allows it
                segments = int((elem['z'][1] - elem['z'][0]) / (args.frames / FPS)) # 25 fps
                for i in range(min(segments, MAX_SEGMENTS)):
                    if len(elem['xy']) == 0:
                        fragments.append({
                            'vid': vid,
                            'vpath': videoPath,
                            'time': [elem['z'][0] + i * (args.frames / FPS), elem['z'][0] + (i+1) * (args.frames / FPS)],
                            'match': elem['match']
                        })

    balanced_fragments = []
    if args.balance:
        print("Classifing fragments in ", len(words), "classes...")
        # Stack of fragments
        class_stacks = [[] for _ in range(len(words))]

        for frag in fragments:
            class_id = frag['match']
            class_stacks[class_id].append(frag)

        print("Stacks:")
        for ind in range(len(class_stacks)):
            print(words[ind], " -> ", len(class_stacks[ind]))

        # Get the minimum number of fragments
        min_fragments = min([len(x) for x in class_stacks])
        min_fragments += 30

        if max_out_frags is not None:
            min_fragments = min(min_fragments, int(max_out_frags/len(words)))

        # Add balanced fragments
        for ind in range(len(class_stacks)):
            stack = class_stacks[ind]
            if random_order:
                random.shuffle(stack)
            balanced_fragments.extend(stack[:min(min_fragments,len(stack))])
    else:
        if max_out_frags is not None:
            fragments = fragments[:max_out_frags]

    fragments = balanced_fragments if args.balance else fragments

    stratify_y = []
    for elem in fragments:
        stratify_y.append(elem['match'])

    train_fragments, test_fragments = train_test_split(fragments, test_size=args.split, random_state=RANDOM_STATE ,stratify=stratify_y)

    total_fragments = len(train_fragments) + len(test_fragments)  

    class_frag_count = [0 for _ in range(len(words))]
    for frag in fragments:
            class_id = frag['match']
            class_frag_count[class_id] += 1
    
    print("Total fragments per class:", class_frag_count)


    count_classes = np.zeros(len(words))
    aug_count_classes = np.zeros(len(words)) 
    t = tqdm(total=total_fragments)
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []
    metasamples_test = []
    metasamples_train = []

    frag_i = 0
    no_flow_counter = 0
    # LOOP AROUND FRAGMENTS
    while frag_i < total_fragments:

        try:

            is_test = False
            if frag_i >= len(train_fragments):
                # TEST
                is_test = True
                frag = test_fragments[frag_i - len(train_fragments)]
            else:
                frag = train_fragments[frag_i]
            
            # Create a VideoCapture object and some useful data
            vid = frag['vid']
            videoPath = frag['vpath']
            cap = cv2.VideoCapture(videoPath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count/fps

            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

            init_frame = int(frag['time'][0] * fps)
            total_final_frame = int(frag['time'][1] * fps)
            final_frame = total_final_frame                                                                                                                                                                                   

            # Discard the video if its too short
            if final_frame - init_frame < args.frames:
                frag_i += 1
                continue

            # Set the initial frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
            
            # Check if camera opened successfully
            if (cap.isOpened()== False): 
                print("Error opening video  file")
                frag_i += 1
                continue

            roi_window = None
            CTTE = 1 # Constat for CTTE * flow

            flow_count = 0
            last_gray_frames = [None, None]
            last_gray_frames_flip = [None, None]

            #Get first frame with considerable optical flow
            not_flow = True
            frame_i = init_frame

            acc_flow = None
            FRAMES_TO_SEARCH = args.frames//2
            for fi in range(FRAMES_TO_SEARCH):
                ret, frameflow1 = cap.read()
                ret, frameflow2 = cap.read()

                frameflow1 = cv2.cvtColor(frameflow1, cv2.COLOR_BGR2GRAY)
                frameflow2 = cv2.cvtColor(frameflow2, cv2.COLOR_BGR2GRAY)

                flowFB = cv2.calcOpticalFlowFarneback(frameflow1, frameflow2, 
                                        None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                #Acc flow
                if fi == 0:
                    acc_flow = flowFB
                else:
                    acc_flow += flowFB

                # Average flow

                fx, fy = flowFB[:,0], flowFB[:,1] 
                v = np.sqrt(fx * fx + fy * fy)
                average = np.average(v[np.nonzero(v)])

                #print("average flow:", average)

                #cv2.imshow("FRAMES",frameflow1)
                #cv2.waitKey(20)

                if average > 0.01 and not_flow:
                    not_flow = False
                    init_frame = frame_i
                    break                

                frame_i = frame_i + 2

            if not_flow:
                frag_i += 1
                continue
            
            #init_frame = frame_i - 2
            final_frame = init_frame + args.frames

            if final_frame - init_frame < args.frames:
                frag_i += 1
                continue

            if final_frame > frame_count:
                frag_i += 1
                continue

            

            # Get roi
            possible_rois = closest_rois(regions[vid], init_frame / fps)
            roi = getMostFlowRoi(possible_rois, cap, init_frame, int(init_frame+(args.frames/2)),padding=args.padding, visualize=VIS)

            if roi is not None:
                x1, y1, x2, y2 = roi
                roi_window = [slice(y1,y1+y2), slice(x1,x1+x2)]
            else:
                print("No ROI")
                frag_i += 1
                continue

            if debug:
                
                img_roi = frameflow1[tuple(roi_window)]
                img_roi = cv2.resize(img_roi,(args.dimension,args.dimension))
                cv2.imshow("ROI", img_roi)

                print("Roi window: ",roi_window)
                print(f"Vid {vid} Frame {frame_i}, End Frame {final_frame}")

                frag_count += 1
                cv2.waitKey(0)
            
            bad = False
            cap.set(cv2.CAP_PROP_POS_FRAMES,init_frame)
            ret, frame = cap.read()
            frameflow1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Initial flow
            
            last_gray_frames[1] = cv2.resize(frameflow1[tuple(roi_window)],(args.dimension,args.dimension))
            last_gray_frames_flip[1] = cv2.resize(cv2.flip(frameflow1[tuple(roi_window)],1),(args.dimension,args.dimension))
        
            sequence = []
            sequence_aug = []
            bad_hog = 0

            # GET FRAMES
            while(not debug and cap.isOpened() and not bad):
                # Capture frame-by-frame
                ret, frame = cap.read()

                if frame_i <= final_frame and ret == True: 
                    
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_frame = cv2.resize(gray_frame[tuple(roi_window)],(args.dimension,args.dimension))

                    # Update Last Frames
                    last_gray_frames[0] = last_gray_frames[1]
                    last_gray_frames[1] = roi_frame
                
                    #Calcular flujo optico
                    flowFB = cv2.calcOpticalFlowFarneback(last_gray_frames[0], last_gray_frames[1], 
                                    None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                    
                    #Normalize flow
                    #TODO

                    if DATA_AUGMENTATION and not is_test:
                        roi_flip_frame = cv2.flip(roi_frame, 1)
                        last_gray_frames_flip[0] = last_gray_frames_flip[1]
                        last_gray_frames_flip[1] = roi_flip_frame
                        flowFB_flip = cv2.calcOpticalFlowFarneback(last_gray_frames_flip[0], last_gray_frames_flip[1], 
                                    None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                    # Visualization
                    if VIS:
                        img_roi = frame[tuple(roi_window)]
                        img_roi = cv2.resize(img_roi,(args.dimension,args.dimension))
                        cv2.imshow("ROI", img_roi)
                        cv2.imshow("Image", frame)
                        cv2.waitKey(100)

                    #Acomular y hacer histograma
                    if not DO_JUST_FLOW:
                        if flow_count == 0:
                            flow = CTTE * flowFB
                            if DATA_AUGMENTATION and not is_test:
                                flow_flip = CTTE * flowFB_flip

                        else:
                            flow += CTTE * flowFB
                            if DATA_AUGMENTATION and not is_test:
                                flow_flip += CTTE * flowFB_flip
                        
                        flow_count = flow_count + 1
                                
                        if flow_count >= FLOW_ACC:
                            flow_count = 0

                            roi_flow = flow
                            modulo, argumento, argumento2 = mi_gradiente(roi_flow)

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

                            #fx, fy = roi_flow[:,0], roi_flow[:,1] 
                            #v = np.sqrt(fx * fx + fy * fy)
                            all_comoponents = orientations * pixels_per_cell[0] * pixels_per_cell[1]
                            count_flow = np.sum([normalized_blocks > 0.5]) / all_comoponents

                            #print("Count flow_roi: ", count_flow)
                            if count_flow <= 0.05:
                                bad_hog = bad_hog + 1
                                if bad_hog > 3:
                                    # print("No flow, skipping")
                                    no_flow_counter += 1
                                    bad = True
                                    break

                            #Add hog features to sequence
                            #print("Len Hog: ",len(normalized_blocks))
                            #print("hog shape: ", normalized_blocks.shape)
                            sequence.append(normalized_blocks)  

                            if DATA_AUGMENTATION and not is_test:
                                roi_flow = flow_flip
                                modulo, argumento, argumento2 = mi_gradiente(roi_flow)
                                normalized_blocks = mi_hog.hog(modulo, argumento2, number_of_orientations=9, pixels_per_cell=(16, 16), 
                                                                        cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False)
                                
                                sequence_aug.append(normalized_blocks)  
                    
                    # Put the flow into the sequence
                    else:
                        
                        count_flow = np.sum([flowFB > 0.5]) / (flowFB.shape[0] * flowFB.shape[1])
                        #print(count_flow)
                        if count_flow <= 0.05:
                            bad_hog = bad_hog + 1
                            if bad_hog > 5:
                                # print("No flow, skipping")
                                no_flow_counter += 1
                                bad = True
                                break

                        if VIS:
                            _,  flow_HSV2 = draw_hsv(flowFB)
                            cv2.imshow('flow', flow_HSV2)  
                        
                        sequence.append(flowFB) 

                        if DATA_AUGMENTATION and not is_test:
                            sequence_aug.append(flowFB_flip) 

                    frame_i =  frame_i + 1
                
                # Break the loop
                else: 
                    break

            # End semi-framgnet processing
            #print("len seq: ", len(sequence))
            if not DO_JUST_FLOW and len(sequence) >= int(args.frames/FLOW_ACC):
                sequence = sequence[:int(args.frames/FLOW_ACC)]
                if DATA_AUGMENTATION and not is_test:
                    sequence_aug = sequence_aug[:int(args.frames/FLOW_ACC)]
            elif DO_JUST_FLOW and len(sequence) >= args.frames:
                sequence = sequence[:args.frames]
                if DATA_AUGMENTATION and not is_test:
                    sequence_aug = sequence_aug[:args.frames]
            else:
                frag_i += 1
                continue

            #print("len seq: ", len(sequence))
            #Sequence and action
            class_id = int(frag['match'])
            count_classes[class_id] += 1

            if is_test:
                ytest.append(class_id)
                Xtest.append(np.array(sequence))
                metasamples_test.append({'vid':vid, 'frames':[init_frame, final_frame], 'roi':roi, 'aug': 0})
            else:
                ytrain.append(class_id)
                Xtrain.append(np.array(sequence))
                metasamples_train.append({'vid':vid, 'frames':[init_frame, final_frame], 'roi':roi, 'aug': 0})

                if DATA_AUGMENTATION and not is_test:
                    ytrain.append(class_id)
                    Xtrain.append(np.array(sequence_aug))
                    aug_count_classes[class_id] += 1                   
                    metasamples_train.append({'vid':vid, 'frames':[init_frame, final_frame], 'roi':roi, 'aug': 1})

            frag_count = frag_count + 1
        
        except Exception as e:
            logging.error(traceback.format_exc())
            print("Continuing...")
        
        cv2.destroyAllWindows()
        cap.release()
        t.update()
        frag_i += 1


    # Export final dataset file

    ytrain = np.array(ytrain)
    Xtrain = np.array(Xtrain)

    ytest = np.array(ytest)
    Xtest = np.array(Xtest)

    # Save dataset
    np.savez(out_file + "_train.npz", a=Xtrain, b=ytrain)
    np.savez(out_file + "_test.npz", a=Xtest, b=ytest)

    # Save meta data
    metadata['samples_train'] = metasamples_train
    metadata['samples_test'] = metasamples_test

    metadata_out = open(metafile, "w")
    json.dump(metadata, metadata_out, indent=0)
    metadata_out.close()

    # Print some info
    print("TRAIN: ")
    print(Xtrain.shape)
    print(ytrain.shape)

    print("TEST: ")
    print(Xtest.shape)
    print(ytest.shape)

    print("Total classes count:", count_classes)
    print("Total augmented classes count:", aug_count_classes)
    print("No flow count: ", no_flow_counter, "(", no_flow_counter/(frag_count+no_flow_counter) ,")")

if __name__ == "__main__":
    main()