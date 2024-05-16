import keras
import theano
from __future__ import print_function
from time import time

import h5py
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split

import numpy as np
import os
import pickle

BATCH_SIZE = 16
FIELD_SIZE = 5 * 300
STRIDE = 1
N_FILTERS = 200

def vectorizeData(text):
    textList = list(text)
    returnList = []
    for item in textList[:1014]:
        returnList.append(ord(item))
    return returnList

validDocsDict = dict()
fileList = os.listdir("BioMedProcessed")
for f in fileList:
    validDocsDict.update(pickle.load(open("BioMedProcessed/" + f, "rb")))

#validDocsDict2 = dict()
#fileList = os.listdir("PubMedProcessed")
#for f in fileList:
#    validDocsDict2.update(pickle.load(open("PubMedProcessed/" + f, "rb")))

print("Loading dataset...")
t0 = time()
documents = []
testPubDocuments = []
allDocuments = []
labels = []
testPubLabels = []
concLengthTotal = 0
discLengthTotal = 0
concCount = 0
discCount = 0
charLength = 1014
charList = []

#combinedDicts = validDocsDict.copy()
#combinedDicts.update(validDocsDict2.copy())

for k in validDocsDict.keys():
    if k.startswith("conclusion") and len(validDocsDict[k]) >= charLength:
        labels.append(0)
        documents.append(vectorizeData(validDocsDict[k]))
        charList.extend(vectorizeData(validDocsDict[k]))
        concCount += 1
        concLengthTotal += len(validDocsDict[k])
    elif k.startswith("discussion") and len(validDocsDict[k]) >= charLength:
        labels.append(1)
        documents.append(vectorizeData(validDocsDict[k]))
        charList.extend(vectorizeData(validDocsDict[k]))
        discCount += 1
        discLengthTotal += len(validDocsDict[k])

charList = set(charList)
        
#for k in validDocsDict2.keys():
#    if k.startswith("conclusion"):
#        testPubLabels.append("conclusion")
#        testPubDocuments.append(vectorizeData(validDocsDict2[k]))
#        concCount += 1
#        concLengthTotal += len(validDocsDict2[k])
#    elif k.startswith("discussion"):
#        testPubLabels.append("discussion")
#        testPubDocuments.append(vectorizeData(validDocsDict2[k]))
#        discCount += 1
#        discLengthTotal += len(validDocsDict2[k])
        
#for k in combinedDicts.keys():
#    if k.startswith("conclusion"):
#        allDocuments.append(vectorizeData(combinedDicts[k]))
#    elif k.startswith("discussion"):
#        allDocuments.append(vectorizeData(combinedDicts[k]))
        
print(len(documents))
print(concLengthTotal * 1.0/ concCount)
print(discLengthTotal * 1.0/ discCount)


train, test, labelsTrain, labelsTest = train_test_split(documents, labels, test_size = 0.95)
test1, test2, labelsTest1, labelsTest2 = train_test_split(test, labelsTest, test_size = 0.9)
print(len(train))
print(len(labelsTrain))

npVecs = np.eye(len(charList))
numToVec = dict()
labelsToVec = dict()
labelsToVec[0] = np.array([1,0])
labelsToVec[1] = np.array([0,1])
counter = 0
for item in charList:
    numToVec[item] = npVecs[counter]
    counter += 1
X_train = np.array([np.array([numToVec[x[y]] for y in x]) for x in train])
Y_train = np.array([np.array(labelsToVec[x]) for x in labelsTrain])
X_test = np.array([np.array([numToVec[x[y]] for y in x]) for x in test1])

X_train.shape
#X_train = np.expand_dims(X_train, axis = 1)

Y_train

# VGG-like convolution stack
model = Sequential()
model.add(Convolution1D(256, 7, border_mode = 'valid', input_shape=(X_train.shape[1], X_train.shape[2]))) 
model.add(Activation('relu'))
model.add(MaxPooling1D(3))

model.add(Convolution1D(256, 7, border_mode = 'valid')) 
model.add(Activation('sigmoid'))
model.add(MaxPooling1D(3))

model.add(Convolution1D(256, 3, border_mode = 'valid')) 
model.add(Activation('relu'))

model.add(Convolution1D(256, 3, border_mode = 'valid')) 
model.add(Activation('sigmoid'))

model.add(Convolution1D(256, 3, border_mode = 'valid')) 
model.add(Activation('relu'))

model.add(Convolution1D(256, 3, border_mode = 'valid')) 
model.add(Activation('sigmoid'))
model.add(MaxPooling1D(3))

model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(2048))
model.add(Dropout(0.5))
model.add(Dense(2))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, nb_epoch=5000, batch_size=BATCH_SIZE, verbose=1, 
          show_accuracy=True, validation_split=0.1)

Y_guess = model.predict_classes(X_test)

numCorrect = 0
for item in range(len(labelsTest1)):
    if Y_guess[item] == labelsTest1[item]:
        numCorrect += 1
print(numCorrect)
print(numCorrect * 1.0 / len(labelsTest1))

