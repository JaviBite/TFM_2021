from re import I
import keras
from keras.models import Sequential

from cv_scripts.libs.mi_hog import normalize

import sys
import json, time

import os
import numpy as np
import cv2

from visualize_dataset import get_hog_image

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, top_k_accuracy_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

def main():

    # Load the model
    model = keras.models.load_model(sys.argv[2])
    model.summary()

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

    if '_train' in file1:
        metadata_file = file1.rsplit('.',maxsplit=1)[0][:-6] + "_metadata.json"
    elif '_test' in file1:
        metadata_file = file1.rsplit('.',maxsplit=1)[0][:-5] + "_metadata.json"
    else:
        metadata_file = file1.rsplit('.',maxsplit=1)[0] + "_metadata.json"
    metadata_in = open(metadata_file,)
    metadata = json.load(metadata_in)

    class_labels = metadata['classes']
    metadata_train_samples = metadata['samples_test']

    jsondata = sys.argv[5]
    with open(jsondata) as json_file:
        data = json.load(json_file)

    videoPaths = data['file']
    local_path = data['config']['file']['loc_prefix']['1'][8:]
    
    acc = int(metadata['config']['accomulation'])
    seg_frames = int(metadata['config']['seg_frames'])
    dim = int(metadata['config']['roi_dim'])

    # Normalice
    X_norm = []

    if sys.argv[3] == '-max':
        for row in range(N_SAMPLES):
            add_samples = []
            for sample in range(N_TIMESTEPS):
                ortientation_hist = X[row,sample,:,:,:]
                normalized_hist = ortientation_hist / np.max(ortientation_hist)
                add_samples.append(normalized_hist)
            X_norm.append(np.array(add_samples))

        X_norm = np.array(X_norm)
        X = X_norm
    elif sys.argv[3] == '-hog':
        X_norm = []
        for row in range(N_SAMPLES):
            add_samples = []
            for sample in range(N_TIMESTEPS):
                ortientation_hist = X[row,sample,:,:,:]
                normalized_hist = normalize(ortientation_hist)
                add_samples.append(normalized_hist)
            X_norm.append(np.array(add_samples))

        X_norm = np.array(X_norm)
    
    # Ravel features into an array
    #X = X.reshape(N_SAMPLES,N_TIMESTEPS,HOG_H*HOG_W*ORIENTATIONS)
    y = []
    
    count_classes = [0] * N_CLASSES
    for yi in labels:
        to_append = np.zeros(N_CLASSES)
        to_append[yi] = 1
        count_classes[yi] += 1
        y.append(to_append)
        
    print("Count classes:",count_classes)

    y = np.array(y).reshape((len(labels), N_CLASSES))

    print("X Shape: ", X_norm.shape)
    print("Y Shape: ", y.shape)

    # evaluate LSTM
    loss, acc = model.evaluate(X_norm, y, verbose=0)
    #print( 'Loss: %f, Accuracy: %f '% (loss, acc*100))

    predict_x =model.predict(X_norm) 
    yhat = np.round(predict_x,decimals=0)

    y_argmax = y.argmax(axis=1)
    yhat_argmax = predict_x.argmax(axis=1)

    # Confusion matrix
    matrix = confusion_matrix(y_argmax, yhat_argmax, normalize='true')

    # Metrics
    f1_s = f1_score(y_argmax, yhat_argmax, average='weighted')
    top_2 = top_k_accuracy_score(y_argmax, predict_x, k=2)

    precision = precision_score(y_argmax, yhat_argmax, average='weighted')
    recall = recall_score(y_argmax, yhat_argmax, average='weighted')

    
    print("Accuracy: ", acc)
    print("TOP_2 accuracy: ", top_2)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_s)

    print(classification_report(y_argmax, yhat_argmax, target_names=class_labels))

    disp_conf = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_labels)
    disp_conf.plot()

    # Training curves

    metrics_file = sys.argv[4]
    metrics_in = open(metrics_file,)
    metrics = json.load(metrics_in)

    history = metrics[0]['history']

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].set_title('Loss')
    axs[0].plot(history['loss'], label='train')
    axs[0].plot(history['val_loss'], label='test')
    axs[0].legend()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')
    axs[1].plot(history['acc'], label='train')
    axs[1].plot(history['val_acc'], label='test')
    axs[1].legend()
    axs[1].set_title('Accuracy')
    axs[1].set_ylabel('Accuracy')

    plt.show()


    # Show bad samples
    bad_samples = y_argmax != yhat_argmax

    out_folder = "out_hogs"
    for i in range(len(bad_samples)):
        if bad_samples[i] and yhat_argmax[i] == 1 and y_argmax[i] == 3:

            predicted = class_labels[yhat_argmax[i]]
            true_label = class_labels[y_argmax[i]]

            print("\nBad sample: ", i)
            print("True label: ", true_label)
            print("Predicted label: ", predicted)
            
            # Load hog sequence
            hog_image_sec = []
            for t in range(N_TIMESTEPS):
                hog = X[i,t,:,:,:]

                hog_image = get_hog_image(hog, (16,16))

                hog_image = (hog_image / np.max(hog_image))  * 255
                hog_image = cv2.cvtColor(hog_image.astype('uint8'),cv2.COLOR_GRAY2BGR)
                hog_image_sec.append(hog_image)

            # Load frame sequence
            row_i = i
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
                roi_frame = cv2.resize(frame[roi_window], (dim,dim))
                frames.append(roi_frame.astype('uint8'))
                frame_i += 1

            next = False
            frame_i = 0
            acc_i = 0
            hog_i = 0
            hog_image = frames[0]*0
            vis_iamge = cv2.vconcat([frames[0],hog_image])
            saving_i = 0
            while not next or saving_i > 0:

                frame = frames[frame_i]
                frame_i = (frame_i + 1) % len(frames)

                if acc_i >= acc:
                    acc_i = 0

                    hog_image = hog_image_sec[hog_i]                    
                    hog_i = (hog_i + 1) % N_TIMESTEPS

                acc_i += 1                

                resized_hog = cv2.resize(hog_image, (dim, dim))

                vis_iamge = cv2.vconcat([frame,resized_hog])
                text = "T: " + true_label + ", P " + predicted
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                vis_iamge = cv2.rectangle(vis_iamge, (0, 0), (w + 4, 14), (255,255,255), -1)
                vis_iamge = cv2.putText(vis_iamge, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                cv2.imshow('Sample', vis_iamge)
                key = cv2.waitKey(50)

                if key == ord('n'):
                    next = True
                if key == ord('s'):
                    print("Saving video...")
                    next = True
                    saving_i = 5 * seg_frames

                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)

                    video_file = out_folder + "/T=" + true_label + "_P=" + predicted + "_I=" + str(i) + ".avi"

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_file, fourcc, 14.0, (dim,dim*2))

                if saving_i > 0:
                    
                    #Save frame
                    out.write(vis_iamge)

                    saving_i -= 1   
                    if saving_i == 0:
                        out.release()                            

                

    return 0


if __name__ == "__main__":
    main()