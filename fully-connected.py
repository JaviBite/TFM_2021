'''
Fully connected neural network for class prediction

@author: Miguel Marcos
'''

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.layers.experimental import RandomFourierFeatures
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD

import bsh_nn_utils as util

num_tiles = 15*15*9 # Number of inputs for the neural network
input_shape = (num_tiles,)

path = 'D:\\GI Lab\\Pruebas_acc_v2\\'
file = 'out_flow_fx_mf500_2c'

num_it = 10

print('Loading dataset...')
X, Y, num_classes = util.load_data(path,file,num_tiles)
print('Total samples: ',X.shape[0])

best_score = 0
best_model = None
best_test_data = None
best_history = None

for i in range(num_it):
    
    print('Iteration ',i, ': Commencing training')
    
    (x_train,y_train),(x_test,y_test) = util.separate_test_data(X,Y,num_classes,num_tiles)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test,  num_classes)

    # Stop training when validation error no longer improves
    earlystop=EarlyStopping(monitor='val_loss', patience=5, 
                            verbose=0, mode='auto')

    # Model definition
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # Model training
    history = model.fit(x_train, y_train,
        batch_size=3,
        epochs=20,
        validation_split=0.2,
        callbacks=[earlystop],
        verbose=False)

    # Model evaluation
    train_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)
    #print('%s %2.2f%s' % ('Accuracy train: ', 100*train_score[1], '%' ))
    print('%s %2.2f%s' % ('Accuracy test:  ', 100*test_score[1], '%'))
    
    if test_score[1] > best_score :
        best_model = model
        best_score = test_score[1]
        best_test_data = (x_test,y_test)
        best_history = history
        print('Updated best model')

(best_x_test,best_y_test) = best_test_data
y_pred = best_model.predict(best_x_test)
util.plot_confusion_matrix(best_y_test,y_pred,num_classes)
util.plot_history(best_history)
