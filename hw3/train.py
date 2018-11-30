
# coding: utf-8

# In[1]:


import os
import numpy as np
import time
import csv
import sys

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, LeakyReLU
import tensorflow as tf 
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

def set_model():
    model = Sequential()
  
    model.add(Conv2D(64, kernel_size=(10, 10), input_shape=(48, 48, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(7, 7), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(7, activation=tf.nn.softmax, kernel_initializer='glorot_normal'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

def set_model_2():
    model = Sequential()
  
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(48, 48, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(7, activation=tf.nn.softmax, kernel_initializer='glorot_normal'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

def read_train(filename):
    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []
  
    file = csv.reader( open(filename, 'r', encoding='big5'), delimiter=",")
    for i, row in enumerate(file):
        if i == 0:
            continue
        if i < 23000:
            onehot = np.zeros((7, ), dtype=np.float)
            onehot[int(row[0])] = 1.0
            train_labels.append(onehot)
            train_data.append(np.fromstring(row[1], dtype=float, sep=' ').reshape((48, 48, 1)))
        else:
            onehot = np.zeros((7, ), dtype=np.float)
            onehot[int(row[0])] = 1.0
            valid_labels.append(onehot)
            valid_data.append(np.fromstring(row[1], dtype=float, sep=' ').reshape((48, 48, 1)))
      
    yield np.asarray(train_data)
    yield np.asarray(train_labels)
    yield np.asarray(valid_data)
    yield np.asarray(valid_labels)
    
def read_test(filename):
  
    test_data = []
  
    file = csv.reader( open(filename, 'r', encoding='big5'), delimiter=",")
    for i, row in enumerate(file):
        if i == 0:
            continue
        test_data.append(np.fromstring(row[1], dtype=float, sep=' ').reshape((48, 48, 1)))
    
    return np.asarray(test_data)
      
def main():
    
    train_data, train_labels, valid_data, valid_labels = read_train(sys.argv[1])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    datagen = ImageDataGenerator(rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.2,
                              zoom_range=[0.8, 1.2],
                              vertical_flip=False, 
                              horizontal_flip=True)
    model_1 = set_model()
    model_2 = set_model_2()
    
    model_1.fit_generator(datagen.flow(train_data, train_labels, batch_size=128), verbose=1,
                        steps_per_epoch=10*len(train_data)//128, epochs = 150,
                        validation_data=(valid_data, valid_labels), callbacks=[early_stopping])
    model_2.fit_generator(datagen.flow(train_data, train_labels, batch_size=128), verbose=1,
                        steps_per_epoch=10*len(train_data)//128, epochs = 150,
                        validation_data=(valid_data, valid_labels), callbacks=[early_stopping])
    
    models=[model_1, model_2]
    
    def ensembleModels(models, model_input):
        # collect outputs of models in a list
        yModels=[model(model_input) for model in models] 
        # averaging outputs
        print("start Average")
        yAvg=Average()(yModels) 
        # build model from same input and avg output
        modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')  
        return modelEns
    
    model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
    modelEns = ensembleModels(models, model_input)
    modelEns.save('trained_model.h5')
main()

