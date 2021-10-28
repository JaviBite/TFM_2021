
import json
from scipy import sparse
import re, random
import cv2
import numpy as np
import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# Ours

from cv_scripts.pot_det import detect_pots
from cv_scripts.pot_det_cv import detect_pots_cv, encuentra_box, iou
from cv_scripts.flow_hog import mi_gradiente
from cv_scripts.libs import mi_hog

import argparse

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

def getROI3(cap, init_frame, coco, n_search_frames, padding, width, height, flow, VIS):

    blurri = 5

    ret, frame = cap.read()

    total_pots = []

    # CV approach
    try:
        pots_cv = detect_pots_cv(cap, init_frame, n_search_frames, blurri, False)
    except OverflowError as of:
        print("After the Overflow error", of, "skipping")
        pots_cv = []

    total_pots = total_pots + pots_cv
    
    # Detectron2 approach
    pots_det = detect_pots_det_n(cap, init_frame, 1, coco)         

    total_pots = total_pots + pots_det

    if VIS:
        for pot in pots_cv:
            cv2.ellipse(frame, pot, (0,0,255), 4)
        for pot in pots_det:
            cv2.ellipse(frame, pot, (255,0,0), 4)
        cv2.imshow("LAS ELIPSES FINALES", frame)
    

    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame) 

    if len(total_pots) > 0:

        # Get the pot with the most flow near it
        most_flow_index = 0
        if len(total_pots) > 1:

            averages = []
            for pot in total_pots:
                c, axes, _ = pot

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
            
        bbox, _ = encuentra_box(total_pots[most_flow_index])
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

def getVidPath(jsondata, localpath, vid):
    namepath = jsondata['file'][vid]['fname']
    return localpath + namepath

def main():

    #Default arguments
    FLOW_STEP = 2
    FLOW_ACC = 4      # 4+1 frames acc

    ROI_DIM = 250
    DET_PAD = 0

    FRAMES_PER_SEQ = 50

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
    parser.add_argument('-aug',"--augmentation", action="store_true", help="Add data augmentation flipping the frames")


    args = parser.parse_args()
    out_file = args.out_file

    max_out_frags = args.max_fragments
    random_order = args.random_order

    VIS = args.visualize

    FLOW_ACC = args.flow_accomulate
    DATA_AUGMENTATION = args.augmentation
    
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

        if final_frame - init_frame < args.frames:
            continue

        if final_frame > frame_count:
            continue

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

        roi_window = None

        CTTE = 1 # Constat for CTTE * flow

        flow_count = 0
        last_gray_frames = [None, None]
        last_gray_frames_flip = [None, None]

        #Get first frame with considerable optical flow

        not_flow = True
        frame_i = init_frame

        acc_flow = None
        FRAMES_TO_SEARCH = 100
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
                init_frame = frame_i - 1
                break                

            frame_i = frame_i + 2

        if not_flow:
            continue
        
        #init_frame = frame_i - 2
        final_frame = init_frame + args.frames

        if final_frame - init_frame < args.frames:
            continue

        if final_frame > frame_count:
            continue

        

        # Get roi
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame - 50)
        ret, frame = cap.read()
        n_search_frames = 5
        #roi = getROI2(cap, init_frame, n_search_frames, args.padding, width, height, acc_flow, VIS)
        
        roi = getROI3(cap, init_frame, coco_predictor, n_search_frames, args.padding, width, height, acc_flow, VIS)

        if roi is not None:
            x1, y1, x2, y2 = roi
            roi_window = [slice(y1,y1+y2), slice(x1,x1+x2)]
        else:
            print("No ROI")
            continue

        
        sequence = []
        sequence_aug = []
        cap.set(cv2.CAP_PROP_POS_FRAMES,init_frame)
        last_gray_frames[1] = cv2.resize(frameflow1[tuple(roi_window)],(args.dimension,args.dimension))
        last_gray_frames_flip[1] = cv2.resize(cv2.flip(frameflow1[tuple(roi_window)],1),(args.dimension,args.dimension))
        bad_hog = 0
        while(cap.isOpened()):
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

                if DATA_AUGMENTATION:
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
                    
                if flow_count == 0:
                    flow = CTTE * flowFB
                    if DATA_AUGMENTATION:
                        flow_flip = CTTE * flowFB_flip

                else:
                    flow += CTTE * flowFB
                    if DATA_AUGMENTATION:
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
                            print("No flow, skipping")
                            break

                    #Add hog features to sequence
                    #print("Len Hog: ",len(normalized_blocks))
                    sequence.append(normalized_blocks)  

                    if DATA_AUGMENTATION:
                        roi_flow = flow_flip
                        modulo, argumento, argumento2 = mi_gradiente(roi_flow)
                        normalized_blocks = mi_hog.hog(modulo, argumento2, number_of_orientations=9, pixels_per_cell=(16, 16), 
                                                                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=VIS)
                        sequence_aug.append(normalized_blocks)  

                frame_i =  frame_i + 1
            
            # Break the loop
            else: 
                break

        cv2.destroyAllWindows()

        # End framgnet processing

        if len(sequence) != int(args.frames/FLOW_ACC):
            continue
        
        #print("len seq: ", len(sequence))
        #Sequence and action
        class_id = frag['class']
        y.append(class_id)
        X.append(np.array(sequence))

        if DATA_AUGMENTATION:
            y.append(class_id)
            X.append(np.array(sequence_aug))

        #print("Len X: ", len(X))

        cap.release()
        frag_count = frag_count + 1
        t.update()

        if max_out_frags is not None and frag_count >= max_out_frags:
            break


    # Write File
    #X = X.reshape(total_fragments, 9*16*16*3*3, 1)

    y = np.array(y)

    X = np.array(X)

    print(X.shape)
    print(y.shape)

    np.savez(out_file + ".npz", a=X, b=y)

if __name__ == "__main__":
    main()