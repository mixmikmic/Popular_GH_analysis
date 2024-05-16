from theano.sandbox import cuda

get_ipython().magic('matplotlib inline')
import utils_modified; reload(utils_modified)
from utils_modified import *
from __future__ import division, print_function

import numpy as np
import random
import sys

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Activation, merge, Flatten, Dropout, Lambda
from keras.layers import LSTM, SimpleRNN
from keras.models import Model, Sequential
from keras.engine.topology import Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import *
from keras.utils.data_utils import get_file

import quandl # pip install quandl
import pandas as pd

# https://keras.io/getting-started/sequential-model-guide/

nbassets = 9

def builder():
    # data array : 20days x 15stocks
    # note that we can name any layer by passing it a "name" argument.
    #main_input = Input(shape=(20,9), dtype='float32', name='main_input')
    
    model = Sequential()
    
    model.add( Dense(output_dim=100, input_shape=(20,nbassets), activation='tanh') )
    
    #model.add( BatchNormalization() )

    # a LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    model.add( SimpleRNN(30,
                         return_sequences=False, stateful=False,
                         activation='relu', inner_init='identity') )
    
    #model.add( Dropout(0.5) )

    model.add( Dense(30, activation='tanh') )
    
    model.add( Dense(1, activation='relu') )
    
    model.compile(optimizer=Adam(1e-3), loss='mean_squared_error')
    
    return model

model1 = builder() # will be trained on simulated data

model1.summary()

if False:
    X = np.random.random((50,20,nbassets))
    Y = np.random.random((50,1))
else:
    AllXs = []
    AllYs = []
    for i in range(1000):
        t = (np.random.rand()-0.5)*6*5 # an offset for the sinus model
        Xs = []
        for stp in range(20):
            vol = 2 + math.sin(t+stp*0.2) # a market volatility
            Xs.append( np.random.randn(1,1,nbassets)*vol ) # one slice of stock returns
        futurevol = 2 + math.sin(t+(20-1+5)*0.2)
        #print(vol, nextvol)
        AllXs.append( np.concatenate(Xs, axis=1) )
        AllYs.append( np.array([futurevol]).reshape((1,1)) )
    X = np.concatenate(AllXs, axis=0)
    Y = np.concatenate(AllYs, axis=0)

model1.fit(X, Y, batch_size=50, nb_epoch=80, validation_split=0.2)

P = model1.predict(X)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.scatter(Y, P)
#plt.plot(Y)
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['without BN','with BN'], loc='upper right')
plt.show()

def qData(tick='XLU'):
    # GOOG/NYSE_XLU.4
    # WIKI/MSFT.4
    qtck = "GOOG/NYSE_"+tick+".4"
    return quandl.get(qtck,
                      start_date="2003-01-01",
                      end_date="2016-12-31",
                      collapse="daily")

'''TICKERS = ['MSFT','JPM','INTC','DOW','KO',
             'MCD','CAT','WMT','MMM','AXP',
             'BA','GE','XOM','PG','JNJ']'''
TICKERS = ['XLU','XLF','XLK','XLY','XLV','XLB','XLE','XLP','XLI']

try:
    D.keys()
except:
    print('create empty Quandl cache')
    D = {}

for tckr in TICKERS:
    if not(tckr in D.keys()):
        print(tckr)
        qdt = qData(tckr)
        qdt.rename(columns={'Close': tckr}, inplace = True)
        D[tckr] = qdt
        
for tck in D.keys():
    assert(D[tck].keys() == [tck])

for tck in D.keys():
    print(D[tck].shape)

J = D[TICKERS[0]].join(D[TICKERS[1]])
for tck in TICKERS[2:]:
    J = J.join(D[tck])

J.head(5)

J.isnull().sum()

J2 = J.fillna(method='ffill')
#J2[J['WMT'].isnull()]

LogDiffJ = J2.apply(np.log).diff(periods=1, axis=0)
LogDiffJ.drop(LogDiffJ.index[0:1], inplace=True)
LogDiffJ.shape

MktData = LogDiffJ.as_matrix(columns=None) # as numpy.array
MktData.shape

model2 = builder() # will be trained on market data

if True:
    AllXs = []
    AllYs = []
    for i in range(500):
        t = np.random.randint(50, MktData.shape[0]-100) # an offset for whole historics
        Xs = []
        for stp in range(20):
            Xs.append( MktData[t+stp,:].reshape(1,1,-1)*100 ) # one slice of stock returns
        futurevol = math.sqrt(np.sum(MktData[t+20:t+20+10,:]*MktData[t+20:t+20+10,:]))*100
        #print(futurevol)
        AllXs.append( np.concatenate(Xs, axis=1) )
        AllYs.append( np.array([futurevol]).reshape((1,1)) )
    X = np.concatenate(AllXs, axis=0)
    Y = np.concatenate(AllYs, axis=0)

X.shape, Y.shape

print(np.min(X), np.mean(X), np.max(X))
print(np.min(Y), np.mean(Y), np.max(Y))

Yc = np.clip(Y, 0, 3*np.mean(Y))
np.mean(Y), np.mean(Yc)

# WARNING : need to scale data to speed up training !!!
factorX = 0.20
factorY = 0.10

# seems like 150 epochs are required to converge : BUG ?
model2.optimizer.lr = 1e-4
model2.fit(factorX*X, factorY*Yc, batch_size=50, nb_epoch=25, validation_split=0.2)

P = model2.predict(factorX*X)/factorY

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.scatter(Yc, P)
#plt.plot(Y)
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
plt.legend(['clipped real vol','model vol'], loc='lower right')
plt.show()



