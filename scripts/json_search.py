
import json
import sys
import codecs, unicodedata
import re
import cv2
import numpy as np

import argparse


def getVidPath(jsondata, localpath, vid):
    namepath = jsondata['file'][vid]['fname']
    return localpath + namepath

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir", type=str, help="Path to the dataset json")
    parser.add_argument('videos_folder', nargs='?', default="none", help="Path to the videos folder (default by dataset json)")
    parser.add_argument("-vis","--visualize", help="Display video fragments",
                    action="store_true")

    args = parser.parse_args()
    
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
    print("Palabra a buscar: ")
    word = input()

    video_ids = {}
    video_data = {}
    for tag, value in zip(data1['metadata'], data1['metadata'].values()):
        #print(tag, value)

        action = str(value['av']['1'])
        vid = int(value['vid'])

        #if word in action and vid not in video_ids:
        if re.search(word,action) is not None:
            if vid not in video_ids:
                video_ids[vid] = []
            
            if action not in video_ids[vid]:
                video_ids[vid].append(action)

            if vid not in video_data:
                video_data[vid] = []

            video_data[vid].append(value)


    videos = dict(sorted(video_ids.items(), reverse=False))
    for vid, value in zip(videos, videos.values()):
        print(vid, value)

    if args.visualize:
        # Display videos
        repVideos = dict(sorted(video_data.items(), reverse=False))

        end = False
        for vid, data in zip(repVideos, repVideos.values()):

            if end:
                break

            videoPath = getVidPath(data1, localpath, str(vid))
            fragments = []
            
            for elem in data:
                if len(elem['z']) == 2:
                    fragments.append({'time':elem['z'], 'act':elem['av']['1']})

            print("Reproducing video: ", videoPath)

            # Create a VideoCapture object and some useful data
            cap = cv2.VideoCapture(videoPath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count/fps

            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

            for frag in fragments:

                init_frame = int(frag['time'][0] * fps)
                final_frame = int(frag['time'][1] * fps)
                #frame_no = init_frame/frame_count

                text = frag['act']

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
                paused = False

                cv2.namedWindow("Video")
                cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
                while(cap.isOpened()):
                    if not paused:
                        # Capture frame-by-frame
                        ret, frame = cap.read()
                        if frame_i <= final_frame and ret == True:
                        
                            # Add some info text
                            pos1 = (int(width*0.05),int(height*0.1))
                            pos2 = (int(width*0.05),int(height*0.95))
                            pos3 = (int(width*0.05),int(height*0.90))
                            cv2.putText(frame, 'Vid: ' + str(vid) + ' Action: ' + text, pos1, cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                            cv2.putText(frame, 'Press q for exit', pos2, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.75, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(frame, 'Press p for pause', pos3, cv2.FONT_HERSHEY_SIMPLEX, 
                            0.75, (0, 0, 255), 2, cv2.LINE_AA)

                            # Display the resulting frame
                            cv2.imshow('Video', frame)
                        
                            
                        
                        # Break the loop
                        else: 
                            break

                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        end = True
                        break
                    elif cv2.waitKey(25) & 0xFF == ord('p'):
                        paused = not paused
                    elif cv2.waitKey(25) & 0xFF == ord('n'):
                        break
                    
                    if not paused:
                        frame_i += 1
                
                if end:
                    break
            
            # When everything done, release 
            # the video capture object
            cap.release()
            
            # Closes all the frames
            cv2.destroyAllWindows()

    # Closing file
    f1.close()

if __name__ == "__main__":
    main()