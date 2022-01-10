import numpy as np
from matplotlib import pyplot as plt
import sys


def main():

    FOLDER = "./out_all_videos_pred"
    Y_FILE = f"{FOLDER}/out_{sys.argv[1]}_y.txt"
    YHAT_FILE = f"{FOLDER}/out_{sys.argv[1]}_yhat.txt"

    # Load data
    y = np.loadtxt(Y_FILE)
    yhat = np.loadtxt(YHAT_FILE)

    yhat = np.matrix(yhat)
    y = np.matrix(y)
    time_frames = np.arange(0, len(y), 1)

    fig, ax = plt.subplots(3,1)
    ax[0].plot(time_frames, yhat[:,0], color = 'green', label = 'Stir')
    ax[0].plot(time_frames, yhat[:,1], color = 'red', label = 'Add')
    ax[0].plot(time_frames, yhat[:,2], color = 'blue', label = 'Flip')
    ax[0].plot(time_frames, yhat[:,3], color = 'yellow', label = 'Others')
    ax[0].set_ylabel("Score")
    ax[0].legend(loc = 'upper left')
    ax[0].set_title("Predictions")

    yhat_thres = yhat
    yhat_thres[yhat > 0.4] = 1
    yhat_thres[yhat <= 0.4] = 0

    ax[1].plot(time_frames, yhat_thres[:,0], color = 'green', label = 'Stir')
    ax[1].plot(time_frames, yhat_thres[:,1], color = 'red', label = 'Add')
    ax[1].plot(time_frames, yhat_thres[:,2], color = 'blue', label = 'Flip')
    ax[1].plot(time_frames, yhat_thres[:,3], color = 'yellow', label = 'Others')
    ax[1].set_ylabel("Score")
    ax[1].set_title("Predictions thresholded")

    ax[2].plot(time_frames, y[:,0], color = 'green', label = 'Stir')
    ax[2].plot(time_frames, y[:,1], color = 'red', label = 'Add')
    ax[2].plot(time_frames, y[:,2], color = 'blue', label = 'Flip')
    ax[2].plot(time_frames, y[:,3], color = 'yellow', label = 'Others')
    ax[2].set_xlabel("Time stamps")
    ax[2].set_ylabel("Score")
    ax[2].set_title("Ground truth")

    plt.show()

    return 0



if __name__ == '__main__':
    main()