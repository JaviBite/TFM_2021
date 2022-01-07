
import cv2
import numpy as np
from skimage.draw import line

# Ours
from cv_scripts.flow_hog import mi_gradiente, draw_hsv
from cv_scripts.libs import mi_hog

# Constants
VID_FILE = "../actions/videos/DCA6323EBFC7_2021_04_16_121056.mp4"
INIT_FRAME = 2719
END_FRAME = 2759
ROI = [slice(28,246), slice(157,394)]
DIM = (250,250)

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

    # Open video
    cap = cv2.VideoCapture(VID_FILE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, INIT_FRAME)
    
    # Get frames flow
    frames_flow = []

    # Initial frame
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_frame = cv2.resize(gray_frame[tuple(ROI)],DIM)

    last_frames = [0,roi_frame]
    frame_i = INIT_FRAME + 1
    while frame_i <= END_FRAME:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_frame = cv2.resize(gray_frame[tuple(ROI)],DIM)

        last_frames[0] = last_frames[1]
        last_frames[1] = roi_frame        

        flowFB = cv2.calcOpticalFlowFarneback(last_frames[0], last_frames[1], 
                    None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                    
        cv2.waitKey(0)

        frames_flow.append(flowFB)
        frame_i += 1

    # HOG configurations
    EXP_1 = {'flow_acc': 4,  'pixels_per_cell': (16, 16)}
    EXP_2 = {'flow_acc': 4,  'pixels_per_cell': (8, 8)}
    EXP_3 = {'flow_acc': 4,  'pixels_per_cell': (32, 32)}

    experiments = [EXP_1, EXP_2, EXP_3]
    hog_secs = [ [] for _ in range(len(experiments)) ]
    for idx, exp in enumerate(experiments):
        print(f"Experiment {idx}")

        frame_i = 0
        flow_count = 0
        flow_acc = np.zeros((DIM[0],DIM[1],2))
        for flow in frames_flow:

            if flow_count >= exp['flow_acc']:
                flow_count = 0
                
                # Calculate HOG
                modulo, _, argumento2 = mi_gradiente(flow_acc)

                pixels_per_cell = exp['pixels_per_cell']

                hog = mi_hog.hog(modulo, argumento2, number_of_orientations=9, pixels_per_cell=pixels_per_cell, 
                                                        cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False)
                hog_secs[idx].append(hog)
                flow_acc = np.zeros((DIM[0],DIM[1],2))
            else:
                flow_acc += flow
                flow_count += 1


        # Visualize
        images = []
        for idx, hog in enumerate(hog_secs[idx]):
            hog_image = get_hog_image(hog, exp['pixels_per_cell'])

            hog_image = (hog_image / np.max(hog_image))  * 255
            hog_image = cv2.cvtColor(hog_image.astype('uint8'),cv2.COLOR_GRAY2BGR)
            images.append(hog_image)

        key = 0
        i = 0
        while key != ord('q'):

            cv2.imshow('HOG', images[i])
            i = (i + 1) % len(images)
            key = cv2.waitKey(1000)

            if key == ord('s'):
                print("Saving hog...")
                cv2.imwrite(f"hog_{pixels_per_cell}_{i}.png", images[4])

        cv2.destroyAllWindows()






    



    return 0


if __name__ == '__main__':
    main()