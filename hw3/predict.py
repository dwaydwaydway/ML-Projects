
# coding: utf-8

# In[1]:


import os
import numpy as np
import time
import csv
import sys

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, LeakyReLU, PReLU
import tensorflow as tf 
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import load_model
def read_test(filename):
  
    test_data = []
  
    file = csv.reader( open(filename, 'r', encoding='big5'), delimiter=",")
    for i, row in enumerate(file):
        if i == 0:
            continue
        test_data.append(np.fromstring(row[1], dtype=float, sep=' ').reshape((48, 48, 1)))
    
    return np.asarray(test_data)
      
def main():
    
    test_data = read_test(sys.argv[1])    
    model = load_model('ensemble.h5')
    predict = model.predict(test_data)
    predict = predict.argmax(axis=-1)
    with open(sys.argv[2], 'w') as f:
        f.write('id,label')
        for i in range(len(predict)):
            f.write('\n' + str(i) + ',' + str(predict[i]))
main()

