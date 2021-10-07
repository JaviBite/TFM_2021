
import json
import sys, os
import codecs, unicodedata
import re, random
import cv2
import numpy as np

import argparse

# Return the region of interest top-left and bottom-right corners
# Call only with the first frame of each fragment, then use cropROI with the returned values.
def getROI(frame, padding, width, height):

    # x1,x2,y1,y2 = funcioncarlos(frame)
    x1,x2,y1,y2 = 10,100,10,100

    if x1-padding<0 : padding=x1
    if x2+padding>=width : padding=int(width-1)-x2
    if y1-padding<0 : padding=y1
    if y2+padding>=height : padding=int(height-1)-y2

    x1,x2,y1,y2 = x1-padding,x2+padding,y1-padding,y2+padding

    return x1,x2,y1,y2

# Crop the region of interest image from a given frame (it corresponds to the pan or cup zone)
# Call once the ROI top-left and bottom-right corners have been returned by getROI in the first frame.
def cropROI(frame,x1,x2,y1,y2):

    ret = frame[y1:y2,x1:x2]

    return ret

def getVidPath(jsondata, localpath, vid):
    namepath = jsondata['file'][vid]['fname']
    return localpath + namepath

def main():

    #Default arguments
    FLOW_STEP = 2
    FLOW_ACC = 4

    ROI_DIM = 250
    DET_PAD = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", type=str, help="Path to the dataset json")
    parser.add_argument('videos_folder', nargs='?', default="none", help="Path to the videos folder (default by dataset json)")
    parser.add_argument("out_file", type=str, default="out.csv", help="Save training data")
    parser.add_argument('-r',"--random_order", action="store_true", help="Random order of video framgets to save")
    parser.add_argument('-mf',"--max_fragments", type=int, default=None, help="Max number of fragments to save")
    parser.add_argument('-p',"--padding", type=int, default=DET_PAD, help="Padding for the dettection zone")
    parser.add_argument('-dim',"--dimension", type=int, default=ROI_DIM, help="Dimenson in pixels of the output square video")

    args = parser.parse_args()
    out_file = args.out_dir

    max_out_frags = args.max_fragments
    random_order = args.random_order
    
    file1 = args.json_dir

    # Opening JSON file
    f1 = open(file1, encoding='utf-8')
    
    # returns JSON object as 
    # a dictionary
    data1 = json.load(f1)

    if args.videos_folder == "none":
        localpath = data1['config']['file']['loc_prefix']['1']
    else:
        localpath = args.videos_folder

    word = 0
    print("Palabras a buscar (separadas por coma): ")
    words = input()

    words = words.split(',')

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

                video_data[vid].append(value)
                break


    videos = dict(sorted(video_ids.items(), reverse=False))
    for vid, value in zip(videos, videos.values()):
        print(vid, value)

    repVideos = dict(sorted(video_data.items(), reverse=False))
    out_name = word.replace(" ","_")
    frag_count = 0

    fragments = []
    for vid, data in zip(repVideos, repVideos.values()):

        videoPath = getVidPath(data1, localpath, str(vid)) 
        for elem in data:
            if len(elem['z']) == 2:
                fragments.append({'time':elem['z'], 'act':elem['av']['1'], 'vpath':videoPath})

    if random_order:
        random.shuffle(fragments)

    fout = open(out_file, "w")

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
        
        # Read until video is completed
        frame_i = init_frame

        # Corners for ROI
        roix1, roix2, roiy1, roiy2 = 0,0,0,0

        sequencia = [[]]
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if frame_i <= final_frame and ret == True:  
                
                #DONE Detectar la ROI una vez para todo el fragmento
                if frame_i == init_frame:
                    roix1,roix2,roiy1,roiy2 = getROI(frame,args.padding,width,height)

                img_roi = cropROI(frame,roix1,roix2,roiy1,roiy2)

                # Visualization
                if False:
                    img_roi = cropROI(frame,roix1,roix2,roiy1,roiy2)
                    img_roi = cv2.resize(to_write,(args.dimension,args.dimension))
                    cv2.imshow("ROI", img_roi)
                    cv2.waitKey()

                #TODO Calcular flujo optico

                #TODO Acomular y hacer histograma

                #TODO Guardar histograma en sequencia

                frame_i =  frame_i + 1
            
            # Break the loop
            else: 
                break

        # End framgnet processing

        #TODO Escribir secuencia y la classe (action)
        sequencia = "TODO"
        fout.write(sequencia + action + "\n")
        
        cap.release()
        frag_count = frag_count + 1

        if max_out_frags is not None and frag_count >= max_out_frags:
            break

        


    # Closing file
    fout.close()
    f1.close()

if __name__ == "__main__":
    main()