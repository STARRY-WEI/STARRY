# -*- coding: utf-8 -*-
import os
import numpy as np
np.random.seed(1337)
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys
from sklearn.externals import joblib
import re
import nltk
from nltk.corpus import wordnet as wn
import copy
import csv


# first, build index mapping words in the embeddings set to their embedding vector


class Data_processing:
    def __init__(self):
        print('Indexing word vectors.')

    def embedding_(self):
        embeddings_index = {}
        f = open(os.path.join('D:\\xiangmu\\data extraction', 'glove.6B.50d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        return embeddings_index

    def read_file(self,filename):
        data_file = pd.read_csv(filename)
        rowDroplist = ['commodity_id','writer', 'date','id']
        data_file.drop(rowDroplist, axis=1, inplace=True)

        return data_file

    # 长度小于3的词以及non - alpha词，小写化
    def CleanWords(self,sentence):
        for idx,val in enumerate(sentence):
            sentence[idx] = re.sub(r'[^\w\s]', '', val)
        stopwords = nltk.corpus.stopwords.words('english')
        cleanWords = []
        for words in sentence:
            if words not in stopwords and len(words) >= 3:
                word_ini = wn.morphy(words)
                if word_ini == None :
                    cleanWords.append(words.lower())
                else:
                    cleanWords.append(word_ini.lower())
        return cleanWords


    # 将单句字符串分割成词,并做清理
    def WordTokener(self, dataframe):
        for index, row in dataframe.iterrows():
            try:
                row['text'] = nltk.word_tokenize(row['text'])
                row['text'] = self.CleanWords(row['text'])
                row['text'] = ' '.join(row['text'])
            except:
                dataframe.drop([index],inplace=True)

        return dataframe.reset_index().drop('index',axis=1)

    def prepare_dataset(self, csv_file):
        print('Processing text dataset')
        texts = []  # list of text samples
        labels = []  # list of label ids
        data = pd.read_csv(csv_file)
        for index, row in data.iterrows():
            if type(row['text']) is str:
                texts.append(row['text'])
                if row['recommend'] == 'Highly Recommend':
                    labels.append(0)
                elif row['recommend'] =='Recommend':
                    labels.append(1)
                elif row['recommend'] =='netural':
                    labels.append(2)
                else:
                    labels.append(3)
        return texts,labels

    def vectorize_textsample(self,texts, labels):
        tokenizer = Tokenizer(nb_words=10000,lower=False)
        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences_generator(texts)
        word_index = tokenizer.word_index
        file = []
        for i in sequences:
            file.append(i)
        data = pad_sequences(file, maxlen=10000, padding='post')
        labels = to_categorical(np.asarray(labels))
        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)
        return data, labels, word_index


class Train_Model:
    def __init__(self):
        print 'begin pre model'


    def split_dataset(self,data, labels):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        # 确定validation的大小
        nb_validation_samples = int(0.2 * data.shape[0])

        X_train = data[:-nb_validation_samples]
        Y_train = labels[:-nb_validation_samples]
        X_val = data[-nb_validation_samples:]
        Y_val = labels[-nb_validation_samples:]

        return X_train,Y_train,X_val,Y_val

    def create_embedding_matrix(self,word_index, embeddings_index):
        nb_words = min(10000, len(word_index))
        embedding_matrix = np.zeros((nb_words+1, 50))
        for word, i in word_index.items():
            if i >= 10000:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i-1] = embedding_vector
        return embedding_matrix

    def create_embedding_layer(self,word_index, embedding_matrix):
        embedding_layer = Embedding(len(word_index) + 1,  # 大或等于0的整数，字典长度，即输入数据最大下标+1，将word_index的维度降为50度
                                    50,  # 大于0的整数，代表全连接嵌入的维度
                                    weights=[embedding_matrix],
                                    input_length=10000,
                                    trainable=False)
        return embedding_layer


    def train_model(self,embedding_layer,X_train,Y_train,X_val,Y_val):
        sequence_input = Input(shape=(10000,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(4, activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                  nb_epoch=2, batch_size=128)

a= Data_processing()
embedding_index = a.embedding_()
# dataframe = a.read_file('asd.csv')
# data = a.WordTokener(dataframe)
# data.to_csv('train')
texts,label = a.prepare_dataset('train')
a.vectorize_textsample(texts,label)
data, labels, word_index = a.vectorize_textsample(texts,label)
b = Train_Model()
X_train,Y_train,X_val,Y_val = b.split_dataset(data,labels)
embedding_matrix = b.create_embedding_matrix(word_index,embedding_index)
embedding_layer = b.create_embedding_layer(word_index,embedding_matrix)
b.train_model(embedding_layer,X_train,Y_train,X_val,Y_val)



