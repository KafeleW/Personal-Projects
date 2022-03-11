#!/usr/bin/env python
# coding: utf-8

# # Final Project: Sentiment Analysis - Twitter Tweets
# # By: Kafele Wimbley

# In[97]:


from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, Bidirectional, Conv1D, Dense, Dropout, Embedding, InputLayer, LSTM, MaxPooling1D, SimpleRNN
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.strings import regex_replace
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as tf
import os
import shutil
import random
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:60% !important; }</style>"))


# In[99]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import string
import collections
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.utils import shuffle
nltk.download('stopwords')
stop_words = stopwords.words("english")


# ### Reading in training data and shuffling 

# In[100]:


# Ths cell reads in data, split data, and shuffle data 
dataTrain = pd.read_csv('twitter_training.csv')

# CSV File did not have a header so I created a new CSV file with header for each column 
headerList = ['id', 'name', 'sentiment', 'tweet']
dataTrain.to_csv("train.csv", header=headerList, index=False)

# Read in data and pop unnecessary attributes 
trainSample = pd.read_csv("train.csv")
trainSample.pop("id")
trainSample.pop("name")

# Using varuable to split data
numRowsTrain = len(trainSample) 

# Grabs all elements after border
trainX = (trainSample['tweet'])[:numRowsTrain]
trainY = (trainSample['sentiment'])[:numRowsTrain]

# Shuffle training data
combined = list(zip(trainX, trainY))
random.shuffle(combined)
trainX[:], trainY[:] = zip(*combined)

trainSample.head()


# In[101]:


for i in range(0, 10):
    print("Sentiment: " + trainY[i])
    print("Tweet: " + trainX[i])


# ### Reading in testing data and shuffling 

# In[102]:


dataTest = pd.read_csv('twitter_test.csv')

# CSV File did not have a header so I created a new CSV file with header for each column 
headerList = ['id', 'name', 'sentiment', 'tweet']
dataTest.to_csv("test.csv", header=headerList, index=False)

# Read in data and pop unnecessary attributes 
testSample = pd.read_csv("test.csv")
testSample.pop("id")
testSample.pop("name")

# numbers of rows to read in from test dataset 
numRowsTest = len(testSample)

# reading in data from test dataset 
testX = (testSample['tweet'])[:numRowsTest]
testY = (testSample['sentiment'])[:numRowsTest]

# shuffle training dataset 
combined = list(zip(testX, testY))
random.shuffle(combined)
testX[:], testY[:] = zip(*combined)

testSample.head()


# In[103]:


for i in range(0, 10):
    print("Sentiment: " + testY[i])
    print("Tweet: " + testX[i])


# ### Converting sentiment string labels to integer labels

# In[104]:


# Function converts sentiment string labels into integers 
def labelInt(labels):
    numLabels = len(labels)
    for i in range(0, numLabels):
        if labels[i] == 'Negative':
            labels[i] = 0
        elif labels[i] == 'Neutral':
            labels[i] = 1 
        elif labels[i] == 'Positive':
            labels[i] = 2
        elif labels[i] == 'Irrelevant':
            labels[i] = 3
    return labels

trainY = labelInt(trainY)
testY = labelInt(testY)


# ### Text preprocessing 

# In[105]:


def textPreproc(textX):
    numText = len(textX)
    for i in range(0, numText):
        
        # replace text with emply string if text value is NaN
        if pd.isna(textX[i]):
            textX[i] = ''
        # Remove numbers 
        textX[i] = re.sub('\w*\d+\w*', '', textX[i])
        # lower case all letters
        textX[i] = textX[i].lower()
        # removing all outside of ascii
        textX[i] = textX[i].encode('ascii', 'ignore').decode()
        # removing '\'
        textX[i] = re.sub("\'\w+", '',textX[i])
        # removing all stop words
        textX[i] = ' '.join([word for word in textX[i].split(' ') if word not in stop_words])
        # remove all punctuations
        textX[i] = re.sub('[%s]' % re.escape(string.punctuation), ' ', textX[i])
        # remove extra spaces
        textX[i] = re.sub('\s{2,}', ' ', textX[i])

    return textX

trainX = textPreproc(trainX)
testX = textPreproc(testX)


# In[106]:


for i in range(0, 10):
    print("Sentiment: " + str(trainY[i]))
    print("Tweet: " + str(trainX[i]))


# In[107]:


for i in range(0, 10):
    print("Sentiment: " + str(testY[i]))
    print("Tweet: " + str(testX[i]))


# ### Numpy array conversion of training/ testing data

# In[160]:


def prepare_data(text, label):
    textX = []
    labelsY = []

    for i in range(0, len(text)):
        textX.append(np.array(text[i]))
        labelsY.append(np.array(label[i]))
    
    return np.array(textX), np.array(labelsY)

trainXA, trainYA = prepare_data(trainX, trainY)
testXA, testYA = prepare_data(testX, testY)

# Print training dataset sentiment amount & shape of training features and labels
print(collections.Counter(trainYA))
print(trainXA.shape, trainYA.shape)

# Print testing dataset sentiment amoun & shape of testing features and labels
print(collections.Counter(testYA))
print(testXA.shape, testYA.shape)


# ### Oversampling of training/ testing data & converting labels to catergorical 

# In[161]:


# reshaping training data inorder to oversample dataset 
feat = trainXA.reshape(-1,1)
label = trainYA.reshape(-1,1)

# oversampling training dataset which will result in equal amount of each type of sentiment and tweets 
oversample = RandomOverSampler(sampling_strategy='all')
xOver, yOver = oversample.fit_resample(feat, label)

# reshaping data again to correct format and converting training labels to categorical 
trainXA = xOver.reshape(-1)
trainYA = yOver.reshape(-1)
trainYA = to_categorical(yOver, 4)

# reshaping testing data inorder to oversample dataset 
testFeat = testXA.reshape(-1, 1)
testLabel = testYA.reshape(-1, 1)

# oversampling testing dataset which will result in equal amount of each type of sentiment and tweets 
testXOver, testYOver = oversample.fit_resample(testFeat, testLabel)

# reshaping data again to correct format and converting training labels to categorical 
testXA = testXOver.reshape(-1)
testYA = testYOver.reshape(-1)
testYA = to_categorical(testYOver, 4)

# Print training dataset sentiment amount & shape of training features and labels
print(collections.Counter(yOver))
print(trainXA.shape, trainYA.shape)

# Print testing dataset sentiment amoun & shape of testing features and labels
print(collections.Counter(testYOver))
print(testXA.shape, testYA.shape)


# ### Benchmark Simple RNN Model

# In[163]:


tf.backend.clear_session()

vectorizeLayer = TextVectorization(output_mode = 'int')

# adapt() fits the TextVectorization layer to our text dataset. This is when the
vectorizeLayer.adapt(trainXA)

modelB = Sequential()

modelB.add(Input(shape=(1,), dtype = 'string'))

# text vectorization layer
modelB.add(vectorizeLayer)

# add an embedding layer to turn integers into fixed-length vectors
modelB.add(Embedding(vectorizeLayer.vocabulary_size(), 128))

# add a fully-connected recurrent layer
modelB.add(SimpleRNN(64))

# add a dense layer
modelB.add(Dense(64, activation = 'relu'))

# add softmax classifier
modelB.add(Dense(4, activation = 'softmax'))

# compiling the model
modelB.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# getting the summary of the model
modelB.summary()


# In[164]:


# epochs used for training
epochs = 10

# training the model
H = modelB.fit(trainXA, trainYA, validation_split = 0.2, epochs = epochs)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, epochs), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, epochs), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, epochs), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, epochs), H.history['val_acc'], label = 'val_acc')
    
# add labels and legend
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()


# In[184]:


print('Test Accuracy')
predictedTestY = modelB.predict(testXA)
predictedTestY = predictedTestY.argmax(axis=1)
testY_class = testYA.argmax(axis=1)
print(classification_report(testY_class, predictedTestY))


# ### Trial 1 Bidirectional Simple RNN

# In[173]:


tf.backend.clear_session()

modelT1 = Sequential()

modelT1.add(Input(shape=(1,), dtype = 'string'))

# text vectorization layer
modelT1.add(vectorizeLayer)

# add an embedding layer to turn integers into fixed-length vectors
modelT1.add(Embedding(vectorizeLayer.vocabulary_size(), 128))

# add a fully-connected recurrent layer
modelT1.add(Bidirectional(SimpleRNN(64)))

# add a dense layer
modelT1.add(Dense(64, activation = 'relu'))

# add softmax classifier
modelT1.add(Dense(4, activation = 'softmax'))

# compiling the model
modelT1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# getting the summary of the model
modelT1.summary()


# In[174]:


# epochs used for training
epochs = 10

# training the model
H = modelT1.fit(trainXA, trainYA, validation_split = 0.2, epochs = epochs)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, epochs), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, epochs), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, epochs), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, epochs), H.history['val_acc'], label = 'val_acc')
    
# add labels and legend
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()


# In[185]:


print('Test Accuracy')
predictedTestY = modelT1.predict(testXA)
predictedTestY = predictedTestY.argmax(axis=1)
testY_class = testYA.argmax(axis=1)
print(classification_report(testY_class, predictedTestY))


# ### Trial 2 Bidirectional Simple RNN with Dropout 

# In[178]:


tf.backend.clear_session()

modelT2 = Sequential()

modelT2.add(Input(shape=(1,), dtype = 'string'))

# text vectorization layer
modelT2.add(vectorizeLayer)

# add an embedding layer to turn integers into fixed-length vectors
modelT2.add(Embedding(vectorizeLayer.vocabulary_size(), 128))

# add a fully-connected recurrent layer
modelT2.add(Bidirectional(SimpleRNN(64)))
modelT2.add(Dropout(0.5))

# add a dense layer
modelT2.add(Dense(64, activation = 'relu'))
modelT2.add(Dropout(0.5))

# add softmax classifier
modelT2.add(Dense(4, activation = 'softmax'))

# compiling the model
modelT2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# getting the summary of the model
modelT2.summary()


# In[179]:


# epochs used for training
epochs = 10

# training the model
H = modelT2.fit(trainXA, trainYA, validation_split = 0.2, epochs = epochs)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, epochs), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, epochs), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, epochs), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, epochs), H.history['val_acc'], label = 'val_acc')
    
# add labels and legend
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()


# In[186]:


print('Test Accuracy')
predictedTestY = modelT2.predict(testXA)
predictedTestY = predictedTestY.argmax(axis=1)
testY_class = testYA.argmax(axis=1)
print(classification_report(testY_class, predictedTestY))


# ### Trial 3 LSTM

# In[181]:


tf.backend.clear_session()

vectorizeLayer = TextVectorization(output_mode = 'int')

vectorizeLayer.adapt(trainXA)

modelT3 = Sequential()

modelT3.add(Input(shape=(1,), dtype = 'string'))

# add layer to the model
modelT3.add(vectorizeLayer)

# add an embedding layer to turn integers into fixed-length vectors
modelT3.add(Embedding(vectorizeLayer.vocabulary_size(), 128))

# add a fully-connected recurrent layer
modelT3.add(LSTM(64))

# add a dense layer
modelT3.add(Dense(64, activation = 'relu'))
modelT3.add(Dropout(0.5))

# add softmax classifier
modelT3.add(Dense(4, activation = 'sigmoid'))

# compiling the model
modelT3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# getting the summary of the model
modelT3.summary()


# In[182]:


# epochs used for training
epochs = 10

# training the model
H = modelT3.fit(trainXA, trainYA, validation_split = 0.2, epochs = epochs)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, epochs), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, epochs), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, epochs), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, epochs), H.history['val_acc'], label = 'val_acc')
    
# add labels and legend
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()


# In[187]:


print('Test Accuracy')
predictedTestY = modelT3.predict(testXA)
predictedTestY = predictedTestY.argmax(axis=1)
testY_class = testYA.argmax(axis=1)
print(classification_report(testY_class, predictedTestY))

