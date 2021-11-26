
from flowGenerator import FlowGenerator

import time, json, numpy as np
import os

from cv_scripts.libs.mi_hog import normalize

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from keras.models import Sequential, Model
from keras.losses import categorical_crossentropy as cc
from keras.regularizers import l1, l2, l1_l2
from keras.layers import LSTM

from keras.layers import Dense, Dropout, Flatten
from keras import layers
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, get
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold

from matplotlib import pyplot

from tqdm import tqdm

def main():

    BATCH_SIZE = 1
    N_CLASSES = 4
    json_filename = "../../BSH_firsthalf_0.2_pots_changes_nogit.json"
    labels = ["remover","poner (?!olla|sarten|cazo)","voltear","^(?!cortar|remover|poner|interaccion|poner|voltear)"]
    labels_alt = ["stir","add","flip","others"]

    out_dir = "../../out_datasets/flow"
    out_dir_test = out_dir + "/test"
    out_dir_val = out_dir + "/val"
    out_dir_train = out_dir + "/train"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for label in labels_alt:
        dir = out_dir + "/" + label
        if not os.path.exists(dir):
            os.makedirs(dir)

    full_generator = FlowGenerator(json_filename, labels, BATCH_SIZE, dimension=100, padding=20, flatten=False,
        frames_sample=25, augmentation=False, balance=True, random_order=False, disbalance_factor=30, max_segments=999)

    trainGenerator, testGenerator, valGenerator = full_generator.get_splits()

    #prgress bar
    pbar = tqdm(total=len(full_generator))

    out_dirs = [out_dir_test, out_dir_val, out_dir_train]
    generators = [trainGenerator, testGenerator, valGenerator]
    for idx in range(len(generators)):

        this_out_dir = out_dirs[idx]

        if not os.path.exists(this_out_dir):
            os.makedirs(this_out_dir)

        metadata = []
        for i in range(len(generators[idx])):
            sample_x, sample_y, meta = full_generator[i]

            label = labels_alt[np.argmax(sample_y)]
            file  = this_out_dir + "/" + label + "/" f'{i:010d}' + ".npz"

            np.savez(file, a=sample_x)
            metadata.append(meta[0])

            pbar.update(1)
        
        pbar.close()

        metadata_out = open(out_dir + "/metadata.json", "w")
        json.dump(metadata, metadata_out, indent=1)
        metadata_out.close()


if __name__ == '__main__':
    main()