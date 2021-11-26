import sys
from random import random
import json
from matplotlib.pyplot import get

from skimage.draw import line
import cv2

from numpy import array
from numpy import cumsum
from numpy import array_equal
import numpy as np

def get_hog_image(hog, pixels_per_cell, mul=255):


    n_cells_row, n_cells_col, number_of_orientations = hog.shape
    c_row, c_col = pixels_per_cell

    s_row, s_col = 250 , 250

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    
    hog_image = np.zeros((n_cells_row, n_cells_col), dtype=float)

    radius = min(c_row, c_col) // 2 - 1
    orientations_arr = np.arange(number_of_orientations)
    # set dr_arr, dc_arr to correspond to midpoints of orientation bins
    orientation_bin_midpoints = (
        np.pi * (orientations_arr + .5) / number_of_orientations)
    dr_arr = radius * np.sin(orientation_bin_midpoints)
    dc_arr = radius * np.cos(orientation_bin_midpoints)
    hog_image = np.zeros((s_row, s_col), dtype=float)
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * c_row + c_row // 2,
                                c * c_col + c_col // 2])
                rr, cc = line(int(centre[0] - dc),
                                int(centre[1] + dr),
                                int(centre[0] + dc),
                                int(centre[1] - dr))
                dato = hog[r, c, o]
                hog_image[rr, cc] += dato

    return hog_image

def main():
    # Load data
    file1 = sys.argv[1]
    files = np.load(file1, allow_pickle=True)
    X, labels = files['a'], files['b']

    jsondata = sys.argv[2]
    with open(jsondata) as json_file:
        data = json.load(json_file)
    
    videoPaths = data['file']
    local_path = data['config']['file']['loc_prefix']['1'][8:]

    N_CLASSES = np.max(labels) + 1
    N_SAMPLES = X.shape[0]
    N_TIMESTEPS = X.shape[1]

    HOG_H = X.shape[2]
    HOG_W = X.shape[3]
    ORIENTATIONS = X.shape[4]

    metadata_file = file1.rsplit('_', maxsplit=1)[0] + "_metadata.json"
    metadata_in = open(metadata_file,)
    metadata = json.load(metadata_in)

    class_labels = metadata['classes']
    metadata_train_samples = metadata['samples_train']

    acc = int(metadata['config']['accomulation'])
    seg_frames = int(metadata['config']['seg_frames'])

    y = []
    
    count_classes = [0] * N_CLASSES
    for yi in labels:
        to_append = np.zeros(N_CLASSES)
        to_append[yi] = 1
        count_classes[yi] += 1
        y.append(to_append)
        
    print("Count classes:",count_classes)

    y = np.array(y).reshape((len(labels), N_CLASSES))

    print("X Shape: ", X.shape)
    print("Y Shape: ", y.shape)

    #Visualizing
    for row_i in range(N_SAMPLES):

        print(metadata_train_samples[row_i])

        initFrame, finalFrame = metadata_train_samples[row_i]['frames']
        initFrame = initFrame - seg_frames
        finalFrame = finalFrame - seg_frames

        vid = str(metadata_train_samples[row_i]['vid'])
        vidPath = local_path + videoPaths[vid]['fname']
            
        class_label = str(class_labels[labels[row_i]])
        
        roi = metadata_train_samples[row_i]['roi']
        x1, y1, x2, y2 = roi
        roi_window = [slice(y1,y1+y2), slice(x1,x1+x2)]

        #Open video
        cap = cv2.VideoCapture(vidPath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, initFrame)

        frames = []
        frame_i = initFrame
        while cap.isOpened() and frame_i <= finalFrame:
            ret, frame = cap.read()
            roi_frame = frame[roi_window]
            frames.append(roi_frame)
            frame_i += 1

        next = False
        frame_i = 0
        acc_i = 0
        hog_i = 0
        while not next:

            frame = frames[frame_i]
            cv2.imshow('ROI: ' + class_label ,frame)

            frame_i = (frame_i + 1) % len(frames)
            acc_i += 1

            if acc_i % acc == 0:
                acc_i = 0
                hog_i = (hog_i + 1) % N_TIMESTEPS

                hog_i = (hog_i + 1) % N_TIMESTEPS
                hog = X[row_i,hog_i,:,:,:]
                hog_image = get_hog_image(hog, (16,16))

                hog_image = hog_image / np.max(hog_image)

                (w, h), _ = cv2.getTextSize(str(row_i), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                hog_image = img = cv2.rectangle(hog_image, (0, 0), (w + 2, 10), (255,255,255), -1)
                hog_image = cv2.putText(hog_image, str(row_i), (2, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                cv2.imshow('HOG: ' + class_label ,hog_image.astype('float32'))

            key = cv2.waitKey(20)

            if key == ord('n'):
                next = True

        


if __name__ == "__main__":
    main()

