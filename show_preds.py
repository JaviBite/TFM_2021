import numpy as np
from matplotlib import pyplot as plt
import sys
import os, re

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, top_k_accuracy_score, accuracy_score
from sklearn.metrics import classification_report


def main():

    FOLDER = "./out_all_videos_pred"
    THRES = 0.4

    # Get all txt files in folder
    yhat_files = [f for f in os.listdir(FOLDER) if re.match(r'.+_yhat.txt', f)]
    y_files = [f for f in os.listdir(FOLDER) if re.match(r'.+_y.txt', f)]

    print(yhat_files)
    print(y_files)

    # Total actions
    y_actions = []
    yhat_actions = []

    for y_file, yhat_file in zip(y_files, yhat_files):
        y = np.loadtxt(f"{FOLDER}/{y_file}")
        ypred = np.loadtxt(f"{FOLDER}/{yhat_file}")
        time_stamps = y.shape[0]

        ypred = np.matrix(ypred)
        #yhat = ypred
        #yhat[yhat > THRES] = 1
        #yhat[yhat <= THRES] = 0
        y = np.matrix(y)

        last_ypre_seq = []
        added = False
        in_action = False
        
        for i in range(time_stamps):
            if np.sum(y[i], axis=1) < 1e-5:
                in_action = False
                
            elif not in_action:

                y_actions.append(y[i])
                in_action = True

            if np.sum(ypred[i], axis=1) < 1e-5 or i == time_stamps - 1:

                if len(last_ypre_seq) > 0:

                    # Append an action decision
                    seq_sum = np.sum(np.matrix(last_ypre_seq), axis=0)/len(last_ypre_seq)
                    seq_sum[seq_sum > THRES] = 1
                    seq_sum[seq_sum <= THRES] = 0
                    yhat_actions.append(seq_sum)

                    last_ypre_seq.clear()

            else:
                last_ypre_seq.append(ypred[i].tolist()[0])


    y_actions = np.squeeze(np.array(y_actions))
    yhat_actions = np.squeeze(np.array(yhat_actions))
    print(y_actions.shape)
    print(yhat_actions.shape)
    for i in range(len(y_actions)):
        print(f"{i}: {y_actions[i]} {yhat_actions[i]}")

    # Metrics
    acc = accuracy_score(y_actions, yhat_actions)
    f1_s = f1_score(y_actions, yhat_actions, average='samples')

    precision = precision_score(y_actions, yhat_actions, average='samples')
    recall = recall_score(y_actions, yhat_actions, average='samples')

    #top_2 = top_k_accuracy_score(y_actions, yhat_actions, k=2)

    print("Accuracy: ", acc)
    #print("top_2: ", top_2)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_s)

    print(classification_report(y_actions, yhat_actions, target_names=["STIR","ADD","FLIP","OTHERS"]))



            




        



    return 0



if __name__ == '__main__':
    main()