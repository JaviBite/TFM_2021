from matplotlib import pyplot as plt
from matplotlib import cm
import json
import sys

def main():

    #load data
    f = open(sys.argv[1],)
    data = json.load(f)

    # Loss
    fig_loss, ax_loss = plt.subplots()
    fig_loss.suptitle('Validation Loss', fontsize=20)

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')

    fig_loss_t, ax_loss_t = plt.subplots()
    fig_loss_t.suptitle('Train Loss', fontsize=20)

    ax_loss_t.set_xlabel('Epoch')
    ax_loss_t.set_ylabel('Loss')

    # Accuracy
    fig_acc, ax_acc = plt.subplots()
    fig_acc.suptitle('Validation Accuracy', fontsize=20)

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')

    fig_acc_t, ax_acc_t = plt.subplots()
    fig_acc_t.suptitle('Training Accuracy', fontsize=20)

    ax_acc_t.set_xlabel('Epoch')
    ax_acc_t.set_ylabel('Accuracy')

    cmap = cm.get_cmap('tab20')

    for indx, experiment in enumerate(data):

        color = cmap(indx/10)
        color2 = cmap(indx/10 + 1/20)

        if 'vis' in experiment:
            label = ''

            for tag in experiment['vis']:
                label += tag + "=" + str(experiment['model'][tag]) + " "

        else:
            drop = experiment['model']['rec_drop']
            l2 = experiment['model']['regularicer']

            label = "drop=" + str(drop) + "l2=" + str(l2)

        history = experiment['history']

        ax_loss_t.plot(history['loss'], label=label, color=color, linewidth=2.0)
        ax_loss.plot(history['val_loss'], label=label, color=color, linewidth=2.0) 

        ax_acc_t.plot(history['acc'], label=label, color=color, linewidth=2.0)
        ax_acc.plot(history['val_acc'], label=label, color=color, linewidth=2.0)  

    ax_acc.legend()
    ax_loss.legend()
    plt.show()


if __name__ == "__main__":
    main()