import sys
sys.path.append("..")

from cv_scripts.flow_hog import mi_gradiente, draw_hsv
import cv2
import numpy as np
import matplotlib as plt



def normalizeFlow(flow):
    print(flow)

    # Change here
    flow[...,0] = cv2.normalize(flow[...,0], None, -1, 1, cv2.NORM_MINMAX)     
    flow[...,1] = cv2.normalize(flow[...,1], None, -1, 1, cv2.NORM_MINMAX)

    # Change here too
    cv2.imshow('Horizontal Component', flow[...,0])
    cv2.imshow('Vertical Component', flow[...,1])

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return flow



def main():

    ctte = 1

    # Read the video
    cap = cv2.VideoCapture("../../actions/videos/DCA6327319E6_2020_07_03_211702.mp4")
    frame = 1325

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)


    if not cap.isOpened():
        print('Unable to open' )
        exit(0)  

    ret, image_np = cap.read()

    #calculate the 50 percent of original dimensions
    width = int(image_np.shape[1])
    height = int(image_np.shape[0])
    # dsize
    dsize = (width, height)

    image_np = cv2.resize(image_np, dsize)
    gray1 = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    ret, image_np = cap.read()    
    image_np = cv2.resize(image_np, dsize)
    gray2= cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)


    flowFB =cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = ctte*flowFB

    roi_flow = flow[80:400, 290:610, :]
    #cv2.imshow('flow HSV', flow_HSV)  

    

    flow_HSV,  flow_HSV2 = draw_hsv(roi_flow)

    cv2.imshow('image', image_np)  
    cv2.imshow('roi', image_np[80:400, 290:610, :])      
    cv2.imshow('roi flow', flow_HSV2) 

    roi_flow_norm = normalizeFlow(roi_flow)
    flow_HSV,  flow_HSV2_norm = draw_hsv(roi_flow_norm)   
    cv2.imshow('roi flow norm', flow_HSV2_norm) 

    cv2.waitKey(0)



    return 0



if __name__ == "__main__":
    main()