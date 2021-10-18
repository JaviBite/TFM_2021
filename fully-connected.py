'''
Fully connected neural network for class prediction
'''

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import numpy as np


num_tiles = 117**2 # Number of inputs for the neural network
classes = [] # (id,classname) pairs will be stored here when reading the metadata file

data_path = 'D:\Miguel\out_flow_f60_mf50.npz'
metadata_path = 'D:\Miguel\out_flow_f60_mf50_metadata.txt'

def load_hog():
    print('Loading dataset...')
    x_train = x_test = np.array([[0]*num_tiles])
    y_train = y_test = ([])

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
        x_test = np.append(x_test,x[:(num_samples//10+1)],axis=0)
        y_test.append(y[:(num_samples//10+1)])
        x_train = np.append(x_train,x[(num_samples//10+1):],axis=0)
        y_train.append(y[(num_samples//10+1):])

    x_test = x_test[1:]
    x_train = x_train[1:]
    print('Done')

    return (x_train,y_train),(x_test,y_test),num_classes

# load HOG
(x_train, y_train), (x_test, y_test), num_classes = load_hog()
print(x_train.shape[0], ' train samples')
print(x_test.shape[0], ' test samples')

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
model.add(Dense(num_classes, activation='softmax', input_shape=(num_tiles,)))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

# Model training
model.fit(x_train, y_train,
    batch_size=5,
    epochs=20,
    validation_split=0.1,
    callbacks=[earlystop],
    verbose=True)

# Model evaluation
train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)
print('%s %2.2f%s' % ('Accuracy train: ', 100*train_score[1], '%' ))
print('%s %2.2f%s' % ('Accuracy test:  ', 100*test_score[1], '%'))

