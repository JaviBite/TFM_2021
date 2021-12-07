
import json
import re, random
import cv2
import numpy as np

import traceback

# import warnings
# warnings.filterwarnings("ignore")

import argparse

def getVidPath(jsondata, localpath, vid):
    namepath = jsondata['file'][vid]['fname']
    return localpath + namepath

def main():

    #Default arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", type=str, help="Path to the dataset json")
    parser.add_argument('videos_folder', nargs='?', default="none", help="Path to the videos folder (default by dataset json)")
    parser.add_argument('-i',"--std_inp", type=str, nargs='?', help="Get the imput from commands instad on the execution")
    parser.add_argument('-r',"--random_order", action="store_true", help="Random order of video framgets to save")

    args = parser.parse_args()
    random_order = args.random_order
    
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

    video_ids = {}
    video_data = {}
    for tag, value in zip(data1['metadata'], data1['metadata'].values()):
        #print(tag, value)

        action = str(value['av']['1'])
        vid = int(value['vid'])

        #if word in action and vid not in video_ids:
        for word in words:
            if re.search(word,action) is not None:

                if vid not in video_data:
                    video_data[vid] = []

                to_append = value
                to_append['match'] = words.index(word)
                video_data[vid].append(to_append)
                break

    repVideos = dict(sorted(video_data.items(), reverse=False))

    # Get the fragments collection
    fragments = []
    for vid, data in zip(repVideos, repVideos.values()):

        videoPath = getVidPath(data1, localpath, str(vid)) 
        for elem in data:
            if len(elem['z']) == 2:
                fragments.append({'time':elem['z'], 'act':elem['av']['1'], 'vpath':videoPath, 'class':elem['match']})

    # Suffle the data if random_order is True
    if random_order:
        random.shuffle(fragments)

    for fragment in fragments:

        time_range = fragment['time']    # array [start_second, end_second]
        action_str = fragment['act']      # string "action noum"
        video_path = fragment['vpath']    # string "path/to/video.mp4"
        class_id = fragment['class']      # int 0,1,2,3...
        class_str = words[class_id]       # string "class_name"

        ############################################
        ## CODIGO DE EJEMPLO                      ##
        ## Abre el video y lo visualiza desde el  ##
        ## inicio hasta el final del fragmento    ##
        ############################################

        # Print info, example
        print("VID = %s, Time range: [%0.3f, %0.3f], class = %s" % (video_path, time_range[0], time_range[1], class_str))

        # Open video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error opening video")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

        init_frame = int(time_range[0] * fps)
        final_frame = int(time_range[1] * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES,init_frame)

        for frame_i in range(init_frame, final_frame):

            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                continue    

            cv2.imshow('Video', frame)
            cv2.waitKey(100)

        ###############################       
        ## FIN CODIGO DE EJEMPLO     ##
        ############################### 

        

        

        

if __name__ == "__main__":
    main()