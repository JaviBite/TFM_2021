
from detectron2 import model_zoo

CONFIG_COCO = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
MODEL_COCO = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

CONFIG_TRACK = "pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml"
MODEL_TRACK = "pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth"

REFRESH_RATE = 60
PAD = 20
INIT_FRAMES = 30

CAT_2_TO_1 = {0: 81, 1: 82, 2: 73, 3: 83, 4: 84}

SCORE_THRESH_TEST = 0.8

import numpy as np, argparse
import time, math
import os, json, cv2, random, torch, sys
from glob import glob

# pysot packages
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.tracker_builder import build_tracker

# Hungarian algorithm
from munkres import Munkres, make_cost_matrix, print_matrix

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes, Instances, pairwise_ioa
from multiTraker import SiamRPNMultiTracker
import time
import tqdm

COLORS = {}

COLORS[73] = (0, 255, 0)        # huevo verde
COLORS[81] = (255, 0, 0)        # patata azul
COLORS[82] = (0, 0, 255)        # patata pelada roja
COLORS[83] = (255, 255, 0)      # llema cian
COLORS[84] = (0, 255, 255)      # cebolla amarilla
COLORS[0]  = (255, 255, 255)    # manos blanco

def detBox_to_Box(box, pad=0):

    box_numpy = np.squeeze(box.tensor.cpu().numpy())
    box_numpy = [int(box_numpy[0]-pad), int(box_numpy[1]-pad), 
                 int(box_numpy[2]-box_numpy[0]+pad), int(box_numpy[3]-box_numpy[1]+pad)]
    
    return box_numpy

def draw_pot_elipse(frame, box):
    ret = frame

    centre = (int(box[0]+box[2]/2),int(box[1]+box[3]/2))

    x_lenght = box[2]/2
    y_lenght = box[3]/2
    axesLength = (int(x_lenght), int(y_lenght))

    ret = cv2.ellipse(ret, centre, axesLength, 0, 0, 360, (0,0,255), 3)
    #ret = cv2.rectangle(ret, (box_np[0], box_np[1]), (box_np[0]+box_np[2], box_np[1]+box_np[3]), (0,255,0), 3)

    return ret

def trackeable(class_id):
    return class_id == 73 or class_id >= 81# or class_id == 0
    #return class_id == 0
    #return class_id == 41 or class_id == 45
    
def costFunc(box1, box2):
    iou = pairwise_ioa(box1, box2)
                        
    max_area = max(box1.area(), box2.area())
    min_area = min(box1.area(), box2.area())
    areaRatio = min_area/max_area
    
    dist = (box2.get_centers() - box1.get_centers())
    dist = np.linalg.norm(dist.cpu().numpy())
    
    #distRatio = dist/MAX_DIST
    
    return ((1-iou + 1-areaRatio)/2 + dist).cpu().numpy()[0][0]

def main():

    REFRESH_RATE = 3
    PAD = 20
    INIT_FRAMES = 30

    SCORE_THRESH_TEST = 0.75
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("in_video", type=str, help="Path to the video that will be processed")
    parser.add_argument("out_video", type=str, nargs="?", default="out.avi", help="Path to save the processed video")

    parser.add_argument("-t", "--threshold", type=float, default=SCORE_THRESH_TEST,
                   help="Threshold for detection")
    
    parser.add_argument("-r", "--refresh_rate", type=int, default=REFRESH_RATE,
                    help="Refresh rate")
                    
    parser.add_argument("-i", "--init_frames", type=int, default=INIT_FRAMES,
                    help="Initialization frames")   

    parser.add_argument("-p", "--padding", type=int, default=PAD,
                    help="Padding for new bounding boxes for tracking initialization") 

    args = parser.parse_args()

    INPUT = args.in_video
    OUTPUT = args.out_video

    REFRESH_RATE = args.refresh_rate
    PAD = args.padding
    INIT_FRAMES = args.init_frames
    SCORE_THRESH_TEST = args.threshold

    # Extract video properties
    video = cv2.VideoCapture(INPUT)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    MAX_DIST = math.sqrt(width*width + height*height)
    
    print("Frames per second: ",frames_per_second)
    REFRESH_RATE = int(3 * frames_per_second)
    print("Refresh rate: ",REFRESH_RATE)

    # Initialize video writer
    video_writer = cv2.VideoWriter(OUTPUT, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

    # Initialize coco predictor
    cfg_coco = get_cfg()
    cfg_coco.merge_from_file(CONFIG_COCO)
    cfg_coco.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_coco.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST  # set threshold for this model
    cfg_coco.MODEL.WEIGHTS = MODEL_COCO
    coco_predictor = DefaultPredictor(cfg_coco)    
    
    #Initialize tracker basestring
    # load config
    cfg.merge_from_file(CONFIG_TRACK)
    cfg.CUDA = torch.cuda.is_available()# and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    print(device)
    
    def runOnVideo(video, maxFrames=None):
        """ Runs the predictor on every frame in the video (unless maxFrames is given),
        and returns the frame with the predictions drawn.
        """
        
        trackers = SiamRPNMultiTracker(MODEL_TRACK)
        new_id = 0
        readFrames = 0
        updateFrames = INIT_FRAMES
        
        # Deleted trackers counters                    
        use = 0
        use2 = 0
        over = 0
        remScore = 0
        
        # Times
        trackTime = 0
        detTime = 0
        CheckTime = 0
        
        CheckDupTime = 0
        initTime = 0
        
        while True:
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            
            vis = frame
            
            # Check trackers outputs           
            track_outs = trackers.track(frame)
            for id in track_outs:
                
                bbox = track_outs[id][0]
                score = track_outs[id][1]
                
                if score >= 0.3:
                    class_id = trackers.getInfo(id)['class_id']
                    #vis = cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 3)
                    #vis = cv2.putText(vis, str(id),(int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)),0, 5e-3 * 200, color,2)
                    #vis = cv2.putText(vis, str(score),(int(bbox[0]+bbox[2]/3), int(bbox[1]+bbox[3]/3)),0, 5e-3 * 150, color,2)

                    vis = draw_pot_elipse(vis,bbox)
                    
            
            #cv2.imshow('frame',vis)
            #cv2.waitKey(0)
            
            #Set refresh frames
            if readFrames % REFRESH_RATE == 0:
                updateFrames = updateFrames + 5                
            
            # Check boxes and init trackers
            if updateFrames > 0:
            
                # Get prediction results for this frame
                coco_outputs = coco_predictor(frame)
                outputs = coco_outputs

                bowls = outputs["instances"][outputs["instances"].pred_classes == 41]
                cups = outputs["instances"][outputs["instances"].pred_classes == 45]

                if len(bowls) > 0:
                    outputs["instances"] = bowls
                    if len(cups) > 0:
                        outputs["instances"] = Instances.cat([bowls["instances"],cups["instances"]])
                elif len(cups) > 0:
                    outputs["instances"] = cups

                fields = outputs["instances"].get_fields()
                classes = fields["pred_classes"]
                
                if len(bowls) > 0 or len(cups) > 0:
                    # Check dup trackers'
                    trackers.checkDups()
                    
                    #Look for new instances to track
                    #Filtring boxes per trackeable class
                    detBoxes = []
                    for i in range(len(fields["pred_boxes"])):
                        detBoxes.append(i)
                        
                    # Detections vs Trackers
                    mat_n = max((len(detBoxes),len(trackers)))
                    cost_matrix = np.ones((len(detBoxes),len(trackers))) * 1
                    m = Munkres()
                    
                    if len(detBoxes) > 0:
                    
                        tstart = time.time()
                        trackIndexToId = {}
                        detIndexToId = {}
                        for i in range(len(detBoxes)):
                            box = fields["pred_boxes"][detBoxes[i]]
                            box = box.to(device)
                            detIndexToId[i] = detBoxes[i]
                            track_i = 0
                            for id in track_outs:
                                trackIndexToId[track_i] = id
                                track_box = trackers.getBox(id)

                                cost = costFunc(track_box, box)
                                cost_matrix[i,track_i] = cost
                                #print("track: ", track_i, " Box: ", i, " -> ", cost)
                                track_i = track_i + 1
                        
                        #print_matrix(cost_matrix, msg='cost matrix')
                        if len(trackers) >= 1:
                            cost_mat_list = cost_matrix.tolist()                
                            indexes = m.compute(cost_mat_list)
                            
                            """print("boxes = ",len(detBoxes))
                            print("Tracks = ",len(trackers))
                            print_matrix(cost_matrix, msg='cost matrix')
                            
                            print(indexes)
                            total_cost = 0
                            for r, c in indexes:
                                x = cost_matrix[r][c]
                                total_cost += x
                                print(('(%d, %d) -> %s' % (r, c, x)))"""
                        
                            #a = 0
                            #input(a)
                            
                            boxesToTack = []
                            #print(trackers)
                            trackedBoxes = []
                            for b, t in indexes:
                                # Box without tracker
                                trackedBoxes.append(b)
                                if (t >= len(trackers)):
                                    boxesToTack.append(detIndexToId[b])
                                elif b < len(detBoxes):
                                    box_id = detIndexToId[b]
                                    box = fields["pred_boxes"][box_id]
                                    init_rect = detBox_to_Box(box,PAD)
                                    
                                    box_t = trackers.getBox(trackIndexToId[t])
                                    box_t = detBox_to_Box(box_t,0)
                                    
                                    """vis = cv2.rectangle(vis, (init_rect[0], init_rect[1]), (init_rect[0]+init_rect[2], init_rect[1]+init_rect[3]), (255,0,0), 3)
                                    vis = cv2.rectangle(vis, (box_t[0], box_t[1]), (box_t[0]+box_t[2], box_t[1]+box_t[3]), (0,255,0), 3)
                                    
                                    vis = cv2.putText(vis, "b: "+str(b),(int(init_rect[0]+init_rect[2]/2), int(init_rect[1]+init_rect[3]/2)),0, 5e-3 * 200, (255,0,0),2)
                                    vis = cv2.putText(vis, "t: "+str(t),(int(box_t[0]+box_t[2]/2), int(box_t[1]+box_t[3]/2)),0, 5e-3 * 200, (0,255,0),2)
                                    """
                                    """if (cost_matrix[b][t] < 15):
                                        #print("Good match")
                                        box = fields["pred_boxes"][box_id]
                                        init_rect = detBox_to_Box(box,PAD)
                                        trackers.update(id, frame, init_rect)"""
                                    
                                    if (cost_matrix[b][t] > 50):
                                        boxesToTack.append(detIndexToId[b])
                            
                            for i in range(len(detBoxes)):
                                if i not in trackedBoxes:
                                    boxesToTack.append(detIndexToId[i])
                                            
                        
                        else:
                            boxesToTack = detBoxes
                            
                        for box_id in boxesToTack:
                            if len(trackers) < 10:
                                box = fields["pred_boxes"][box_id]
                                init_rect = detBox_to_Box(box,PAD)
                                trackers.addTracker(frame,init_rect,box, readFrames,{"class_id": int(classes[box_id].cpu())})
                                
                                vis = cv2.rectangle(vis, (init_rect[0], init_rect[1]), (init_rect[0]+init_rect[2], init_rect[1]+init_rect[3]), (0,0,255), 3)
                                vis = cv2.putText(vis, "add b",(int(init_rect[0]+init_rect[2]), int(init_rect[1]+init_rect[3]/2)),0, 5e-3 * 200, (0,0,255),2)
                                
                        tend = time.time()
                        initTime = initTime + (tend - tstart)
                        
                        """vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                        vis = v.draw_instance_predictions(vis, outputs["instances"].to("cpu"))
                        vis = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)
                        
                        cv2.imshow('frame',vis)
                        cv2.waitKey(0)  """ 

                        #cv2.imshow('frame',vis)
                        #cv2.waitKey(0)               
                        
                    trackers.removeBadTrackers()
            
            # Make sure the frame is colored
            #vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            # Draw a visualization of the predictions using the video visualizer
            #visualization = v.draw_instance_predictions(vis, outputs["instances"].to("cpu"))

            # Convert Matplotlib RGB format to OpenCV BGR format
            #visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

            #yield visualization
            yield vis

            updateFrames -= 1
            if updateFrames < 0:
                updateFrames = 0
                
            readFrames += 1
            if maxFrames is not None and readFrames > maxFrames:
                break
           
        print("==================")
        print("Removed trackers:")
        print("\tUse: ", use, "\n\tUse2: ", use2, "\n\tOver: ", over)
        print("\tScore: ", remScore)
        
        trackTime = trackTime / num_frames
        detTime = detTime / num_frames
        CheckTime = CheckTime / num_frames
        
        CheckDupTime = CheckDupTime / (num_frames/REFRESH_RATE)
        initTime = initTime / (num_frames/REFRESH_RATE)
        
        print("Time usage average:")
        print("\tTracking: ", trackTime, "\n\tDetection: ", detTime,
              "\n\tChecking: ", CheckTime, "\n\tCheckingDups: ", CheckDupTime,
              "\n\tInit Trackers: ", initTime)

    # Enumerate the frames of the video
    for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames, position=0, leave=True):

        # Write test image
        #cv2.imwrite('POSE detectron2.png', visualization)

        # Write to video file
        video_writer.write(visualization)

    # Release resources
    video.release()
    video_writer.release()
    
    
if __name__ == '__main__':
    main()