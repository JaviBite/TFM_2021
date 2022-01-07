
import json, random, string, math, re
import argparse
import cv2, codecs
from tqdm import tqdm
import numpy as np

from cv_scripts.pot_det import detect_pots
from cv_scripts.pot_det_cv import detect_pots_cv, iou, encuentra_box

import warnings
warnings.filterwarnings("ignore")

# Initialize coco predictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torch.cuda import is_available

CONFIG_COCO = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
MODEL_COCO = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
SCORE_THRESH_TEST = 0.2

# Elipse angle = 0
def normalize_elipse(elipse):
    c, (w,h), angle = elipse
    th = angle * math.pi/180

    xa =  np.sqrt( w*w * math.cos(th)**2 + h*h * math.sin(th)**2 )/2
    ya =  np.sqrt( w*w * math.sin(th)**2 + h*h * math.cos(th)**2 )/2

    return (c, (xa,ya), 0)

def elip_to_viaElip(elipse):
    e = [0,0,0,0,0]
    c, (w,h), angle = elipse

    e[0] = 4 # 4 means ellipse
    e[1], e[2] = c
    e[3] = w
    e[4] = h
    
    return e

def det2_to_elip(pots):
    ret_pots = []
    for pot in pots:
        x1, y1, x2, y2 = pot
        c = (x1+int(x2/2), y1+int(y2/2))
        e = (c,(x2,y2),0)
        ret_pots.append(e)
    
    return ret_pots

def elip_to_box(elip):
    c, (w,h), angle = elip
    x1 = c[0] - w/2
    y1 = c[1] - h/2
    x2 = c[0] + w/2
    y2 = c[1] + h/2
    return (x1,y1,x2,y2)

def filter_pots(current_pots, new_pots):
    new_filtered_pots = []

    # We add all cv pots
    for pot_d in new_pots:
        b1 = elip_to_box(pot_d)
        add = True

        for pot_c in current_pots:
            b2 = elip_to_box(pot_c)
            eiou = iou(b1,b2)

            # If the iou is high enough, we discard the box
            if eiou >= 0.5:
                add = False
                break
        
        if add:
            new_filtered_pots.append(pot_d)
    
    return current_pots + new_filtered_pots

def detect_pots_det_n(cap, init_frame, n_frames, coco_pred):

    total_pots = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame) 

    for i in range(n_frames):
        ret, frame = cap.read()

        if not ret:
            break

        pots = detect_pots(frame, coco_pred, 0)
        pots = det2_to_elip(pots)
    
        total_pots = filter_pots(total_pots, pots)
        

    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame) 

    return total_pots

# Return the time in seconds where the pot is ready to be dettected
MOTION_ACTION_REG = "poner (olla|sarten|cazo)|quitar (olla|sarten|cazo)|^mover (olla|sarten|cazo)"
def get_motion_fragments(data):

    fragments = {}

    for value in data['metadata'].values():

        vid = int(value['vid'])

        action = None
        if '1' in value['av'] and len(value['xy']) == 0:
            action = str(value['av']['1'])
            hit = re.search(MOTION_ACTION_REG,action)
            
        if action is not None and hit is not None:

            if vid not in fragments:
                fragments[vid] = []

            fragments[vid].append(value['z'][1])

    return fragments

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("via_json", type=str, help="Path to the json of the dataset (via annotations)")
    parser.add_argument('videos_dir', nargs='?', default=None, help="Dir where the videos are, if not set, json dir will be used")
    parser.add_argument('out_json', nargs='?', default="out.json", help="Path file for the output json")
    parser.add_argument('-t','--time', type=float, default=None, help="Sample every t seconds")
    parser.add_argument('-v','--verbose', help="Print progress info on console", action="store_true")
    parser.add_argument('-vis','--visualice', help="Show dettections on screen", action="store_true")
    parser.add_argument('-sf','--search_frames', type=int, default=10, help="Search frames looking for pans")
    parser.add_argument('-mv','--max_videos', type=int, default=None, help="Number of videos to process")

    return parser.parse_args()
  

def main():

    args = parseargs()

    file_ours = args.via_json
    videos_dir = args.videos_dir
    outfile = args.out_json
    verbose = args.verbose
    vis = args.visualice
    sampling_time = args.time

    n_search_frames = args.search_frames
    max_videos = args.max_videos

    # Opening JSON file
    fours = open(file_ours, encoding='utf-8')
    data_ours = json.load(fours)
    fours.close()

    # Coco predictor
    cfg_coco = get_cfg()
    cfg_coco.merge_from_file(CONFIG_COCO)
    cfg_coco.MODEL.DEVICE = 'cuda' if is_available() else 'cpu'
    cfg_coco.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST  # set threshold for this model
    cfg_coco.MODEL.WEIGHTS = MODEL_COCO
    coco_predictor = DefaultPredictor(cfg_coco)

    if videos_dir is None:
        videos_dir = data_ours['config']['file']['loc_prefix']['1'][8:]

    total_videos = len(data_ours['file'])

    if max_videos is None:
        max_videos = total_videos

    # Get motion fragments
    fragments = get_motion_fragments(data_ours)
        

    # Counters
    modified_videos = []
    count_videos = 0

    #Progress bar
    t = tqdm(total=max_videos, unit='video')

    # Open each video
    all_ok = True
    for value in data_ours['file'].values():
        video_path = videos_dir + value['fname']
        vid = int(value['fid'])
        
        if verbose: print("Video: ", video_path)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video: ", video_path)
            continue

        # Get video info
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #Initial values
        init_frame = 50
        frag_i = 0
        finished_fragments = False
        finished_temporal_sampling = False
        while (not finished_temporal_sampling or not finished_fragments):

            if finished_temporal_sampling:
                if (vid in fragments and frag_i < len(fragments[vid])):
                    init_frame = int((fragments[vid][frag_i] + 1) * video_fps)
                    frag_i += 1
                else:
                    finished_fragments = True

            # Process the init frames and detect the pots

            # Discard fragments at the end of the video
            if(init_frame > video_length - video_fps * 1):
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
            ret, frame = cap.read()
            if not ret:
                print("Error reading frames on ", video_path)
                all_ok = False
                break

            # CV approach
            pots_cv = []
            try:
                pots_cv = detect_pots_cv(cap, init_frame, n_search_frames, blurri=5, VIS=False)
            except OverflowError as of:
                print("After the Overflow error", of, "skipping")
            
            # Detectron2 approach
            pots_det = detect_pots_det_n(cap, init_frame, n_search_frames, coco_predictor) 

            # Add the detectron 2 videos if no overlap cv ones
            total_pots = filter_pots(pots_cv, pots_det)

            if vis:

                frame_copy = frame.copy()

                for pot in pots_cv:
                    cv2.ellipse(frame, pot, (0,0,255), 4)
                for pot in pots_det:
                    cv2.ellipse(frame, pot, (255,0,0), 4)

                for pot in total_pots:
                    cv2.ellipse(frame_copy, pot, (0,255,0), 4)
                
                cv2.imshow("Elipses", frame)
                cv2.imshow("Total elipses", frame_copy)
                cv2.waitKey(0)

            if len(total_pots) > 0:

                if verbose: print("\tAdding new metadata...")

                for pot in total_pots:
                    
                    # Get the [angle,x0,y0,w,h] format
                    norm_pot = normalize_elipse(pot)
                    via_pot = elip_to_viaElip(norm_pot) 

                    # Generating the action value
                    time = 0.0
                    if init_frame > 50:
                        time = init_frame / video_fps

                    new_value = {'vid':str(vid), 
                                'flg':0, 
                                'z':[time], 
                                'xy':via_pot, 
                                'av':{"2":"0"}}

                    # Generating random metadata id
                    new_key = None
                    while(True):
                        randid = ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=8))
                        new_key = str(vid) + '_' + randid
                        if new_key not in data_ours['metadata']:
                            break

                    # Inserting the new data
                    data_ours['metadata'][new_key] = new_value

                if vid not in modified_videos:
                    modified_videos.append(vid)

            # No pots detected
            else:

                if verbose: print("\tNo pots detected")
            

            # Increment the initial frame
            if sampling_time is None:
                finished_temporal_sampling = True
            else:
                init_frame += int(sampling_time * video_fps)
                if (init_frame + n_search_frames + 5 * video_fps >= video_length):
                    finished_temporal_sampling = True

        count_videos += 1
        t.update()

        if count_videos >= max_videos:
            break

    t.close()
    print("\n")

    # Default replace dir
    #for i in ["1","2","3","4"]:
    #    data_ours['config']['file']['loc_prefix'][i] = "file:///E:/tfm/actions/videos" #"file:///%DIR_TO_VIDEOS%"  #TODO
    
    # Add the new atributte
    attribute = {"aname":"pot","anchor_id":"FILE1_Z1_XY1","type":3,"desc":"Recipiente","options":{"0":"sarten","1":"cacerola"},"default_option_id":"0"}
    data_ours['attribute']['2'] = attribute

    if all_ok:
        if verbose: print("Dumping data...")

        with codecs.open(outfile, 'w', encoding='utf-8') as out:
            json.dump(data_ours, out, indent=None, ensure_ascii=False)

            # Closing file
            out.close()

        print("All ok!")
        print("Ratio of modified videos: ", len(modified_videos)/max_videos)

    else:
        print("Some error ocurred")

if __name__ == "__main__":
    main()