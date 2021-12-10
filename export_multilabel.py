import json, argparse, os, sys, re, random

import cv2

from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

import traceback, logging

from keras.utils.all_utils import to_categorical

# Ours
from cv_scripts.pot_det import detect_pots
from cv_scripts.pot_det_cv import detect_pots_cv, encuentra_box, iou
from cv_scripts.flow_hog import mi_gradiente, draw_hsv
from cv_scripts.libs import mi_hog

def overlapping(time1, time2, threshold=1):

    delta = 0
    if time1[0] < time2[1] and time2[0] < time1[1]:
        latest_start = max(time1[0], time2[0])
        earliest_end = min(time1[1], time2[1])
        delta = earliest_end - latest_start


    return delta > threshold

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
    parser.add_argument('-b',"--balance", action="store_true", help="Force balance between all the classes")
    parser.add_argument('-w',"--window", type=float, default=1.0, help="Percetnage of new data for the new window of fragment.")
    parser.add_argument('-s',"--split", type=float, default=0.2, help="Split percentage for the dataset in Train and Test")
    parser.add_argument('-d',"--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()
    OUT_FILE = args.out_file
    VID_FOLDER = args.videos_folder

    FRAMES_PER_SEQ = args.frames
    MAX_FRAGS = args.max_fragments
    RANDOM_ORDER = args.random_order

    VIS = args.visualize or args.debug
    DEBUG = args.debug

    DET_PAD = args.padding
    ROI_DIM = args.dimension
    FLOW_ACC = args.flow_accomulate

    MAX_SEGMENTS = args.max_segments
    
    JFILE = args.json_dir

    # Opening JSON file
    f1 = open(JFILE, encoding='utf-8')
    
    # returns JSON object as 
    # a dictionary
    data1 = json.load(f1)
    f1.close()

    if VID_FOLDER == "none":
        localpath = data1['config']['file']['loc_prefix']['1'][8:]
    else:
        localpath = args.videos_folder
    
    words = args.std_inp
    if words is None:
        print("Palabras a buscar (separadas por coma): ")
        words = input()

    words = [x.strip() for x in words.split(',')]
    print("Classes: ", words)

    N_CLASSES = len(words)

    # Save metadata
    metafile = OUT_FILE.split('.')[0] + "_metadata.json"

    metadata = {}

    metadata['classes'] = words

    metadata['config'] = {'padding': DET_PAD,
                          'fragments': -1,
                          'seg_frames': FRAMES_PER_SEQ,
                          'accomulation': FLOW_ACC,
                          'random': RANDOM_ORDER,
                          'roi_dim': ROI_DIM}

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

            regions[vid].append(value)

    # Short regions in chronological order
    for vid in regions.keys():
        regions[vid] = sorted(regions[vid], key=lambda d: d['z'][0], reverse=False)

    repVideos = dict(sorted(video_data.items(), reverse=False))

    fragments = {}
    for vid, data in zip(repVideos, repVideos.values()):

        videoPath = getVidPath(data1, localpath, str(vid)) 
        len_data = len(data)
        fragments[vid] = []
        for i in range(len_data):
            elem = data[i]
            if len(elem['z']) == 2: # Temporal data

                # Temporal segment if the fragment duration allows it
                num_segments = int((elem['z'][1] - elem['z'][0]) / (FRAMES_PER_SEQ / FPS)) # 25 fps
                for k in range(min(num_segments, MAX_SEGMENTS)):
                    
                    time = [elem['z'][0] + k * (FRAMES_PER_SEQ / FPS), elem['z'][0] + (k+1) * (FRAMES_PER_SEQ / FPS)]
                    match = to_categorical(elem['match'], num_classes=N_CLASSES)

                    if len(elem['xy']) == 0:    # No region data
                        fragments[vid].append({
                            'vid': vid,
                            'vpath': videoPath,
                            'time': time,
                            'match': to_categorical(elem['match'], num_classes=N_CLASSES)
                        })

    # Check multi label
    ml_fragments = []
    overlapping_count = 0
    overlapping_classes_count = np.zeros(N_CLASSES)
    for vid in fragments.keys():

        n_videos = len(fragments[vid])
        overlapped = []
        for i in range(n_videos):
            if i not in overlapped:
                overlapped.append(i)
                elem1 = fragments[vid][i]

                time1 = elem1['time']

                # Check if the temporal segment is in the same video
                for j in range(n_videos):
                    if j not in overlapped:
                        elem2 = fragments[vid][j]

                        time2 = elem2['time']

                        if overlapping(time1,time2):
                            overlapped.append(j)

                            # Add multi label
                            new_match = elem1['match'] + elem2['match']
                            if np.max(new_match) == 1:
                                elem1['match'] = new_match
                                # print("OVERLAPPING: ", elem1['match'])
                                # print("\tVID: ", vid)
                                # print("\tTIME: ", time1, time2)

                                overlapping_count += 1
                                overlapping_classes_count += new_match


                ml_fragments.append(elem1)
    
    print("Overlapping count: ", overlapping_count)
    print("Overlapping classes count: ", overlapping_classes_count)




    balanced_fragments = []
    if args.balance:
        print("Classifing fragments in ", N_CLASSES, "classes...")
        # Stack of fragments
        class_stacks = [[] for _ in range(N_CLASSES)]
        class_count = [0] * N_CLASSES

        for frag in ml_fragments:
            idx = 0
            for class_id in frag['match']:
                if class_id == 1:
                    if sum(frag['match']) > 1:
                        class_stacks[idx].insert(0,frag)
                    else:
                        class_stacks[idx].append(frag)
                    class_count += frag['match'] 
                    break
                idx += 1

        print("Stacks:")
        for ind in range(N_CLASSES):
            print(words[ind], " -> ", class_count[ind])

        # Get the minimum number of fragments
        min_fragments = int(min(class_count))
        min_fragments += 30 # Add a lil disbalance for more samples

        if MAX_FRAGS is not None:
            min_fragments = min(min_fragments, int(MAX_FRAGS/N_CLASSES))

        # Add balanced fragments
        for ind in range(len(class_stacks)):
            stack = class_stacks[ind]
            if RANDOM_ORDER:
                random.shuffle(stack)
            balanced_fragments.extend(stack[:min(min_fragments,len(stack))])
    else:
        if MAX_FRAGS is not None:
            fragments = fragments[:MAX_FRAGS]

    fragments = balanced_fragments if args.balance else fragments

    train_fragments, test_fragments = train_test_split(fragments, test_size=args.split, random_state=RANDOM_STATE)

    total_fragments = len(train_fragments) + len(test_fragments)  

    train_frag_count = [0] * N_CLASSES
    test_frag_count = [0] * N_CLASSES

    for frag in train_fragments:
            train_frag_count += frag['match']
    for frag in test_fragments:
            test_frag_count += frag['match']
    
    print("Train classes count:", train_frag_count)
    print("Test classes count:", test_frag_count)

    overlapped_samples_train = 0
    for frag in train_fragments:
        if sum(frag['match']) > 1:
            overlapped_samples_train += 1

    overlapped_samples_test = 0
    for frag in test_fragments:
        if sum(frag['match']) > 1:
            overlapped_samples_test += 1

    print("Overlapped samples train:", overlapped_samples_train)
    print("Overlapped samples test:", overlapped_samples_test)

    count_classes = np.zeros(len(words)) 
    t = tqdm(total=total_fragments)

    Xtrain = []
    ytrain = []

    Xtest = []
    ytest = []

    metasamples_test = []
    metasamples_train = []

    frag_i = 0
    no_flow_counter = 0
    frag_count = 0
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

            if DEBUG:
                
                img_roi = frameflow1[tuple(roi_window)]
                img_roi = cv2.resize(img_roi,(args.dimension,args.dimension))
                cv2.imshow("ROI", img_roi)

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
            while(not DEBUG and cap.isOpened() and not bad):
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

                    # Visualization
                    if VIS:
                        img_roi = frame[tuple(roi_window)]
                        img_roi = cv2.resize(img_roi,(args.dimension,args.dimension))
                        cv2.imshow("ROI", img_roi)
                        cv2.imshow("Image", frame)
                        cv2.waitKey(100)

                    #Acomular y hacer histograma
                    if flow_count == 0:
                        flow = CTTE * flowFB

                    else:
                        flow += CTTE * flowFB
                    
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

                    frame_i =  frame_i + 1
                
                # Break the loop
                else: 
                    break

            # End semi-framgnet processing
            #print("len seq: ", len(sequence))
            if len(sequence) >= int(args.frames/FLOW_ACC):
                sequence = sequence[:int(args.frames/FLOW_ACC)]
            else:
                frag_i += 1
                continue

            #print("len seq: ", len(sequence))
            #Sequence and action
            count_classes += frag['match']

            if is_test:
                ytest.append(frag['match'])
                Xtest.append(np.array(sequence))
                metasamples_test.append({'vid':vid, 'frames':[init_frame, final_frame], 'roi':roi, 'aug': 0})
            else:
                ytrain.append(frag['match'])
                Xtrain.append(np.array(sequence))
                metasamples_train.append({'vid':vid, 'frames':[init_frame, final_frame], 'roi':roi, 'aug': 0})

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
    np.savez(OUT_FILE + "_train.npz", a=Xtrain, b=ytrain)
    np.savez(OUT_FILE + "_test.npz", a=Xtest, b=ytest)

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

    y_test_ml = 0
    y_train_ml = 0
    for yi in ytest:
        if sum(yi) > 1:
            y_test_ml += 1
    for yi in ytrain:
        if sum(yi) > 1:
            y_train_ml += 1
    
    print("TEST multi label samples: ", y_test_ml)
    print("TRAIN multi label samples: ", y_train_ml)

    print("Total classes count:", count_classes)
    print("No flow count: ", no_flow_counter, "(", no_flow_counter/(frag_count+no_flow_counter) ,")")



if __name__ == '__main__':
    main()