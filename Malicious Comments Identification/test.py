import os
import csv
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import pickle
from gensim.models.word2vec import Word2Vec
import jieba
from keras.layers import Embedding, Dense, LSTM, GRU, Dropout, Flatten, Activation, CuDNNGRU
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.losses import *
from keras.models import load_model
max_length = 100
import sys

def read_data(test_x):
    test_data = []
    with open(test_x, 'r', encoding='utf-8') as file:
        reader = csv.reader( file, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            test_data.append(row[1].replace(" ", ""))
    
    return np.asarray(test_data)
    
def split_word(train_data, dic, cut_type):
    train_data = list(train_data)
    jieba.set_dictionary(dic)
    
    for i in range(len(train_data)):
        train_data[i] = list(jieba.cut(train_data[i], cut_all=cut_type))
    return np.asarray(train_data)

def trim(data):
    count = 0
    for i in range(len(data)):
        if len(data[i]) > max_length:
            data[i] = data[i][:max_length]
            
def tokenize(data, w2v_index):
    tokenized = []
    for l, sentance in enumerate(data):
        temp = []
        for word in sentance:
            try:
                temp.append(w2v_index[word])
            except:
                temp.append(0)
        tokenized.append(temp)
        
    return np.asarray(tokenized)

def set_model(embedding_matrix):
    
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      trainable=False,
                      input_length=max_length))
    
    model.add(GRU(20, return_sequences=True, dropout=0.5))    
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', kernel_initializer='glorot_normal'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    
    test_data = read_data(sys.argv[1])
    
    test_data = split_word(test_data, sys.argv[2], False)
    
    w2v_model = Word2Vec.load("word2vec.model")
    embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
    w2v_index = {}
    vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vector = vocab
        embedding_matrix[i + 1] = vector
        w2v_index[word] = i + 1 

    trim(test_data)
    
    test_x = pad_sequences(tokenize(test_data, w2v_index), maxlen=max_length)
    
    model = load_model('best.h5')
    
    predict = model.predict(test_x)
    predict = predict.argmax(axis=-1)
    with open(sys.argv[3], 'w') as f:
        f.write('id,label')
        for i in range(len(predict)):
            f.write('\n' + str(i) + ',' + str(predict[i]))
main()