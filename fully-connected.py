'''
Fully connected neural network for class prediction

@author: Miguel Marcos
'''

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD

import bsh_nn_utils as util

num_tiles = 15*15*9 # Number of inputs for the neural network
input_shape = (num_tiles,)

path = 'D:\\GI Lab\\Pruebas_acc_v2\\'
file = 'out_flow_f8_mf500'

num_it = 10

print('Loading dataset...')
Xtrain,Xtest,Ytrain,Ytest,num_classes,classes = util.load_data(path,file,num_tiles)
print('Total samples: ',Xtrain.shape[0]+Xtest.shape[0])

total_score = 0

# convert class vectors to binary class matrices
Ytrain = keras.utils.to_categorical(Ytrain, num_classes)
Ytest  = keras.utils.to_categorical(Ytest,  num_classes)

# Stop training when validation error no longer improves
earlystop=EarlyStopping(monitor='val_loss', patience=3, 
                        verbose=0, mode='auto')

for i in range(num_it):
    # Model definition
    model = Sequential()
    #model.add(BatchNormalization())
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    #model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1024, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    
    # Model training
    history = model.fit(Xtrain, Ytrain,
        batch_size=3,
        epochs=20,
        validation_split=0.2,
        callbacks=[earlystop],
        verbose=False)

    # Model evaluation
    train_score = model.evaluate(Xtrain, Ytrain, verbose=0)
    test_score = model.evaluate(Xtest, Ytest, verbose=0)
    #print('%s %2.2f%s' % ('Accuracy train: ', 100*train_score[1], '%' ))
    print('%s %2.2f%s' % ('Accuracy test:  ', 100*test_score[1], '%'))
    total_score = total_score + test_score[1]
    
    Ypred = model.predict(Xtest)
    util.plot_confusion_matrix(Ytest,Ypred,num_classes)
#util.plot_history(history)
print('%s %2.2f%s' % ('Averages accuracy:  ', 100*total_score/num_it, '%'))
