'''
Fully connected neural network for class prediction
'''

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD
import numpy as np


num_tiles = 117**2 # Number of inputs for the neural network
classes = [] # (id,classname) pairs will be stored here when reading the metadata file

path = 'D:\\GI Lab\\'
data_path = path+'out_test_aug_500_1_p10.npz'
metadata_path = path+'out_test_aug_500_1_p10_metadata.txt'

def load_hog():
    print('Loading dataset...')
    x_train = x_test = np.array([[0]*num_tiles])
    y_train = y_test = np.array([])

    print('Opening metadata file...')
    md_file = open(metadata_path)
    num_classes = 0
    for line in md_file:
        classes.append((num_classes,line[3:]))
        num_classes = num_classes + 1
    print('Read ',num_classes,' classes')
    md_file.close()

    print('Loading training data...')
    data = np.load(data_path)

    print('Separating test data...')
    for (i,_) in classes:
        x = np.squeeze(data['a'][:])
        y = data['b'][:]
        x = x[y==i]
        y = y[y==i]
        num_samples = len(y)
        test_samples = num_samples//10+1
        train_samples = num_samples-test_samples
        print('Class: ',i)
        print('Train samples: ',train_samples,', Test samples: ', test_samples)
        x_test = np.append(x_test,x[:test_samples],axis=0)
        y_test = np.append(y_test,y[:test_samples])
        x_train = np.append(x_train,x[test_samples:],axis=0)
        y_train = np.append(y_train,y[test_samples:])

    x_test = x_test[1:]
    x_train = x_train[1:]
    print('Done')

    return (x_train,y_train),(x_test,y_test),num_classes

# load HOG
(x_train, y_train), (x_test, y_test), num_classes = load_hog()
print('Total train samples: ',x_train.shape)
print('Total test samples: ',x_test.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)

# random permutation of training data
np.random.seed(0)
p = np.arange(x_train.shape[0])
np.random.shuffle(p)
x_train = x_train[p]
y_train = y_train[p]

# Stop training when validation error no longer improves
earlystop=EarlyStopping(monitor='val_loss', patience=5, 
                        verbose=1, mode='auto')

# Model definition
model = Sequential()

# Single layer perceptron
model.add(Dense(512, activation='relu', input_shape=(num_tiles,)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Model training
model.fit(x_train, y_train,
    batch_size=5,
    epochs=20,
    validation_split=0.1,
    callbacks=[earlystop],
    verbose=False)

# Model evaluation
train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)
print('%s %2.2f%s' % ('Accuracy train: ', 100*train_score[1], '%' ))
print('%s %2.2f%s' % ('Accuracy test:  ', 100*test_score[1], '%'))

