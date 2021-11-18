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

    for i in range(len(bad_samples)):
        if bad_samples[i]:

            predicted = class_labels[yhat_argmax[i]]
            true_label = class_labels[y_argmax[i]]

            print("\nBad sample: ", i)
            print("True label: ", true_label)
            print("Predicted label: ", predicted)

            hog_image_sec = []
            for t in range(N_TIMESTEPS):
                hog = X[i,t,:,:,:]

                hog_image = get_hog_image(hog, (16,16))

                hog_image = hog_image / np.max(hog_image)
                hog_image_sec.append(hog_image.astype('float32'))

            t = 0
            key = 0
            while (key != ord('n')):
                hog_image = hog_image_sec[t]
                (w, h), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                hog_image = img = cv2.rectangle(hog_image, (0, 0), (w + 2, 10), (255,255,255), -1)
                hog_image = cv2.putText(hog_image, str(i), (2, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                cv2.imshow('HOG: ' , hog_image)
                key = cv2.waitKey(100)
                t = (t + 1) % N_TIMESTEPS

            key = cv2.waitKey()
            
            if key == ord('s'):

                print("Saving video...")
                                
                out_folder = "out_hogs"
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)


                #Save video
                video_file = out_folder + "/T=" + true_label + "_P=" + predicted + "_I=" + str(i) + ".avi"

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_file, fourcc, 7.0, hog_image.shape)

                for j in range(len(hog_image_sec)*5):
                    j = j % N_TIMESTEPS
                    image = np.uint8(hog_image_sec[j]*255)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)    
                    out.write(image)

                out.release()

    return 0


if __name__ == "__main__":
    main()