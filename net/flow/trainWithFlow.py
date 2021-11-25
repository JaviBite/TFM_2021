
from flowLoader import FlowLoader

import time, json, numpy as np

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

def create_model(num_classes, input_shape, lstm_units, rec_dropout, lstm_act, lstm_rec_act, final_act, hidden_act, dropouts, hidden_dense_untis, regu):

    model = Sequential()
    model.add(Dropout(dropouts[0], input_shape=input_shape)) # (n_timesteps, n_features)
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=False, activation=lstm_act, recurrent_activation=lstm_rec_act, recurrent_dropout=rec_dropout, \
                             activity_regularizer = l1(0.01))))# \
                             #kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))))
    model.add(Dropout(dropouts[1]))
    model.add(Flatten())
    model.add(Dense(hidden_dense_untis, activation = hidden_act, kernel_regularizer=l1(l1=0.001)))
    model.add(Dropout(dropouts[2]))
    model.add(Dense(num_classes, activation = final_act))

    return model

def create_ConvModel(num_classes, input_shape, lstm_units, rec_dropout, lstm_act, lstm_rec_act, final_act, hidden_act, dropouts, hidden_dense_untis, regu):

    inp = layers.Input(shape=input_shape)

    x = layers.Dropout(dropouts[0])(inp)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation=lstm_act,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation=lstm_act,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        #return_sequences=True,
        activation=lstm_act,
    )(x)

    x = Flatten()(x)

    x = layers.Dropout(dropouts[1])(x)
    x = layers.Dense(hidden_dense_untis, activation= hidden_act)(x)
    x = layers.Dropout(dropouts[2])(x)
    x = layers.Dense(num_classes, activation= final_act)(x)

    # Next, we will build the complete model and compile it.
    model = Model(inp, x)

    return model

def main():

    BATCH_SIZE = 1
    N_CLASSES = 4
    json_filename = "../../out_datasets/flow"
    labels = ["stir","add","flip","others"]

    full_generator = FlowLoader(json_filename, labels, BATCH_SIZE)

    print(len(full_generator))

    print(full_generator[0])

def main2():

    BATCH_SIZE = 1
    N_CLASSES = 4
    json_filename = "../../BSH_firsthalf_0.2_pots_changes_nogit.json"
    labels = ["remover","poner (?!olla|sarten|cazo)","voltear","^(?!cortar|remover|poner|interaccion|poner|voltear)"]

    full_generator = FlowGenerator(json_filename, labels, BATCH_SIZE, dimension=100, padding=20, flatten=False,
        frames_sample=25, augmentation=False, balance=True, random_order=False, disbalance_factor=30, max_segments=999)

    trainGenerator, testGenerator, valGenerator = full_generator.get_splits()

    sampleX, _ = trainGenerator[0]
    print(sampleX.shape)
    # define problem
    n_timesteps =  sampleX.shape[1]
    n_sequences =  sampleX.shape[0]
    n_features = sampleX.shape[2:]

    INPUT_SHAPE = (n_timesteps, n_features[0], n_features[1], n_features[2])

    SPLITS = 5
    
    lr = [0.001] * SPLITS
    lstm_units = [32] * SPLITS
    rec_drop = [0.3] * SPLITS
    lstm_act = ['tanh'] * SPLITS
    lstm_rec_act = ['hard_sigmoid'] * SPLITS
    final_act = ['softmax'] * SPLITS
    hidden_act = ['sigmoid'] * SPLITS
    dropouts = [[0.5,0.4,0.3]] * SPLITS
    hidden_dense_untis = [64] * SPLITS
    regularicer = [0.001] * SPLITS

    optimizers = ['adam'] * SPLITS
    losses = ['categorical_crossentropy'] * SPLITS
    epochs = [30] * SPLITS
    
    to_vis = ['regularicer','rec_drop']
    i = 0

    # Define the K-fold Cross Validator
    #kfold = StratifiedKFold(n_splits=SPLITS, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    model = None
    models_metrics = []
    #for train, val in kfold.split(X, labels):
    for i in range(SPLITS):

        model = create_ConvModel(N_CLASSES, INPUT_SHAPE, lstm_units[i], rec_drop[i], lstm_act[i], 
                            lstm_rec_act[i], final_act[i], hidden_act[i], dropouts[i], hidden_dense_untis[i], regularicer[i])
        opt = get(optimizers[i])
        opt.learning_rate = lr[i]
        model.compile(loss= losses[i] , optimizer= opt , metrics=[ 'acc' ])
        #print(model.summary())

        # Early Estopping
        es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5)

        start = time.time()
        
        history = model.fit_generator(generator=trainGenerator,
                    validation_data=valGenerator, epochs=epochs[i], verbose=1, shuffle=True, callbacks=[es, reduce_lr])

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_time = {'hours': hours, 'minutes': minutes, 'seconds': seconds}

        fold_no = fold_no + 1

        print("Elapsed time: ", "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

        model_json = {'lr': lr[i],
                    'lstm_units': lstm_units[i],
                    'rec_drop': rec_drop[i],
                    'lstm_act': lstm_act[i],
                    'lstm_rec_act': lstm_rec_act[i],
                    'final_act': final_act[i],
                    'hidden_act': hidden_act[i],
                    'dropouts': dropouts[i],
                    'hidden_dense_untis': hidden_dense_untis[i],
                    'optimizers': optimizers[i],
                    'losses': losses[i],
                    'epochs': epochs[i],
                    'regularicer': regularicer[i]
        }

        metrics = history.history

        lr_list = []
        for f in metrics['lr']:
            lr_list.append(float(f))
        metrics['lr'] = lr_list
        models_metrics.append({'model': model_json, 'history': metrics, 'etime': elapsed_time, 'vis':to_vis})
        
        i += 1

    # dumps results
    out_file = open("out_model_metrics.json", "w")
    json.dump(models_metrics, out_file, indent=1)

    model.save('out_model_bilstm.h5')

    # evaluate LSTM
    #X, y = get_sequences(100, n_timesteps, size_elem)
    loss, acc = model.evaluate_generator(valGenerator, verbose=1)
    print( 'Loss: %f, Accuracy: %f '% (loss, acc*100))

    predict_x=model.predict_generator(valGenerator, verbose=1)
    yhat=np.round(predict_x,decimals=0)

    fig, axs = pyplot.subplots(2, 1, constrained_layout=True)
    axs[0].set_title('Loss')
    axs[0].plot(history.history['loss'], label='train')
    axs[0].plot(history.history['val_loss'], label='test')
    axs[0].legend()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')
    axs[1].plot(history.history['acc'], label='train')
    axs[1].plot(history.history['val_acc'], label='test')
    axs[1].legend()
    axs[1].set_title('Accuracy')
    axs[1].set_ylabel('Accuracy')


    # Confusion matrix

    y_true = [np.argmax(valGenerator[i][1], axis=1) for i in range(BATCH_SIZE)]

    matrix = confusion_matrix(
    np.concatenate(y_true),    
    np.argmax(model.predict_generator(valGenerator, steps=BATCH_SIZE), axis=1), normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot()
    
    pyplot.show()

    # Loss histogram
    losses = []
    for i in range(len(yhat)):
        losses.append(cc(y_true[i], yhat[i]))

    fig, axs = pyplot.subplots(1, 1, constrained_layout=True)
    axs.hist(losses)
    axs.set_title('Loss histogram')
    axs.set_xlabel('Loss value')
    axs.set_ylabel('Frecuency')
    
    pyplot.show()


if __name__ == '__main__':
    main()