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

    N_CLASSES = np.max(labels) + 1
    N_SAMPLES = X.shape[0]
    N_TIMESTEPS = X.shape[1]

    HOG_H = X.shape[2]
    HOG_W = X.shape[3]
    ORIENTATIONS = X.shape[4]

    metadata_file = file1.split('.')[0] + "_metadata.json"
    metadata_in = open(metadata_file,)
    metadata = json.load(metadata_in)

    class_labels = metadata['classes']

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

        for hog_i in range(N_TIMESTEPS):
            
            hog = X[row_i,hog_i,:,:,:]
            hog_image = get_hog_image(hog, (16,16))

            hog_image = hog_image / np.max(hog_image)

            class_label = str(class_labels[labels[row_i]])
            (w, h), _ = cv2.getTextSize(str(row_i), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            hog_image = img = cv2.rectangle(hog_image, (0, 0), (w + 2, 10), (255,255,255), -1)
            hog_image = cv2.putText(hog_image, str(row_i), (2, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
            cv2.imshow('HOG: ' + class_label ,hog_image.astype('float32'))
            cv2.waitKey(20)
        
        cv2.waitKey(100)


if __name__ == "__main__":
    main()

