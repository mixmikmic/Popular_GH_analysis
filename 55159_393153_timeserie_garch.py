from theano.sandbox import cuda

get_ipython().magic('matplotlib inline')
from __future__ import division, print_function

import math
import numpy as np
import random
import sys

from numpy.random import normal
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Embedding, Activation, merge, Flatten, Dropout, Lambda
from keras.layers import LSTM, SimpleRNN, TimeDistributed
from keras.models import Model, Sequential
from keras.layers.merge import Add, add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.constraints import nonneg
from keras.layers.convolutional import *
from keras import backend as K
from keras.utils.data_utils import get_file

look_back = 12
batch_size = 1

def mllklh(args): # minus log-likelihood of gaussian
    var_t, eps2_t = args
    return 0.5*math.log(2*math.pi) + 0.5*K.log(var_t)  + 0.5*(eps2_t/var_t)

#len(Xs), Xs[0].shape, modelUNR.predict([X[10:15,0] for X in Xs]).shape

MatPlus = np.zeros((look_back-1,look_back))
MatPlus[:,1:] = np.eye(look_back-1)
#print(MatPlus)

MatMinus = np.zeros((look_back-1,look_back))
MatMinus[:,:-1] = np.eye(look_back-1)
#print(MatMinus)

Dplus = Dense(look_back-1,
              weights=[MatPlus.T, np.zeros((look_back-1,))],
              input_dim=look_back)
Dplus.trainable = False

Dminus = Dense(look_back-1,
              weights=[MatMinus.T, np.zeros((look_back-1,))],
              input_dim=look_back)
Dminus.trainable = False

I = Input(batch_shape=(batch_size, look_back,1), dtype='float32')
I2 = Lambda(lambda x : K.square(x), output_shape=(look_back,1))(I)
rnn = SimpleRNN(return_sequences=True, unroll=False,
                units=1, input_shape=(look_back, 1),
                bias_constraint=nonneg(), # insure positive var
                kernel_constraint=nonneg(), # insure positive var
                recurrent_constraint=nonneg(), # insure positive var
                activation=None,
                stateful=True)
O1 = rnn(I2)
O1f = Flatten()(O1)
O1m = Dminus(O1f)

V = Lambda(lambda x : K.sqrt(x), output_shape=(look_back,))(O1f) # get volatility

I2f = Flatten()(I2)
I2p = Dplus(I2f)

Errors = Lambda(mllklh, output_shape=(look_back-1,))([O1m,I2p])

Error = Lambda(lambda x : K.sum(x, axis=1), output_shape=(look_back-1,))(Errors)

modelT = Model(inputs=I, outputs=Errors) # training model

def special_loss(dummy, errorterms):
    return errorterms

modelT.compile(optimizer='adadelta', loss=special_loss)

modelV = Model(inputs=I, outputs=V) # simulation model

Dplus.get_weights()[0].shape, Dplus.get_weights()[1].shape

I._keras_shape, I2._keras_shape, O1._keras_shape, O1f._keras_shape, V._keras_shape, Errors._keras_shape, Error._keras_shape

onearr = np.ones((batch_size, look_back, 1)).astype('float32')
print(I2.eval({I:onearr}).shape)
print(O1.eval({I:onearr}).shape)
print(O1f.eval({I:onearr}).shape)
print(V.eval({I:onearr}).shape)
print(Errors.eval({I:onearr}).shape)
print(Error.eval({I:onearr}).shape)
print(modelT.predict(onearr).shape)
print(modelV.predict(onearr).shape)

rnn(I2).eval({I:onearr}) # dry run to allow weight setting
rnn.set_weights([np.array([[0.5]]),np.array([[1]]),np.array([1.5])])
print( O1f.eval({I:onearr}) )
print( O1m.eval({I:onearr}) )

#print( I2f.eval({I:onearr}) )
#print( I2p.eval({I:onearr}) )

Error.eval({I:onearr}).shape

modelT.summary()

kappa = 0.000003
alpha = 0.85
beta = 0.10
lvar = kappa / (1-alpha-beta)
print(math.sqrt(lvar)*math.sqrt(255))

math.sqrt(lvar) # standard deviation of simulated data set

F = 1/math.sqrt(lvar) # will have to scale training data by F and Kappa by F^2 (alpha and beta unchanged)

rnn(I2).eval({I:onearr}) # dry run to allow weight setting
rnn.set_weights([np.array([[beta]]),np.array([[alpha]]),np.array([kappa])*F*F])

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, allow_overlap=True):
    dataX, dataY = [], []
    if allow_overlap:
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    else:
        # non overlap
        for i in range(0, int(dataset.shape[0]/batch_size)*batch_size-look_back, look_back):
            #print(i)
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

train = []
trainvars = []
var_t = lvar
for t in range(250*8):
    eps = math.sqrt(var_t) * normal()
    var_t = kappa + alpha * var_t + beta * (eps*eps)
    train.append(eps) # percent
    trainvars.append(var_t)
train = np.array(train).reshape(-1,1)
trainX, trainY = create_dataset(train, look_back, allow_overlap=False)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#print(trainX, trainY)

trainX.shape, trainY.shape, trainX.transpose((0,2,1)).shape

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(np.cumsum(train))
plt.subplot(2,1,2)
plt.plot(np.sqrt(trainvars)*math.sqrt(255))
plt.show()

var_t = 0#lvar
check_vars = []
#for eps in train[0:(50*look_back),0]:
#    var_t = kappa + alpha * var_t + beta * (eps*eps)
for k, eps in enumerate(train[(50*look_back):((50*look_back)+3*batch_size*look_back),0]):
    var_t = kappa + alpha * var_t + beta * (eps*eps)
    check_vars.append(var_t)
    if k<3*look_back:
        print(math.sqrt(var_t*255))
        if ((k>0) and ((k+1) % look_back == 0)):
            print('-------', k, '--------')

rnn.states[0].get_value().shape, rnn.states[0].get_value()[:,0]*math.sqrt(255)/F

trainX.transpose((0,2,1)).shape

A = trainX.transpose((0,2,1))[50,:,0]
B = train[(50*look_back):((50*look_back)+1*look_back),0]

var_t = 0

print( math.sqrt((kappa + alpha * var_t + beta * (A[0]**2))*255) )
print( math.sqrt((kappa + alpha * var_t + beta * (B[0]**2))*255) )

np.sqrt(np.array(check_vars)*255)[0], np.squeeze(Vs.reshape((1,-1)), axis=0)[0]

rnn.reset_states()

#Vs = V.eval({I:trainX.transpose((0,2,1))[100:(100+batch_size),:,:].astype('float32')*F})*math.sqrt(255)/F
Vs0 = modelV.predict( trainX.transpose((0,2,1))[50:(50+batch_size),:,:].astype('float32')*F )*math.sqrt(255)/F
Vs1 = modelV.predict( trainX.transpose((0,2,1))[51:(51+batch_size),:,:].astype('float32')*F )*math.sqrt(255)/F
Vs2 = modelV.predict( trainX.transpose((0,2,1))[52:(52+batch_size),:,:].astype('float32')*F )*math.sqrt(255)/F
Vs = np.vstack([Vs0,Vs1,Vs2])
#np.squeeze(Vs.reshape((1,-1)))#[0:20,0], Vs[0,:], Vs[1,:]
#Vs[0:3,:]
plt.figure(figsize=(10,7))
plt.plot( np.sqrt(np.array(check_vars)*255), c='red', marker='+' )
plt.plot( np.squeeze(Vs.reshape((1,-1)), axis=0), c='black' )

test_arr = trainX.transpose((0,2,1))[50:(50+batch_size),:,:].astype('float32')
ErrorStart = np.sum(Errors.eval({I:test_arr*F }))
print(ErrorStart)

print( modelT.predict(trainX.transpose((0,2,1))[50:(50+batch_size),:,:]*F)[0] )
print('-------------------------')
print( modelV.predict(trainX.transpose((0,2,1))[50:(50+batch_size),:,:]*F)[0]*math.sqrt(255)/F )

max_batches = int(trainX.transpose((0,2,1)).shape[0]/batch_size)
max_batches

modelT.optimizer.lr.set_value(1e-1) # 1e-3 seems too large !
#modelT.optimizer.lr.get_value()

Ydummy = trainX.transpose((0,2,1))
#print(Ydummy.shape)
#print( trainX.transpose((0,2,1)).shape )
hist0 = modelT.fit(trainX.transpose((0,2,1))[0:(max_batches*batch_size),:,:]*F,
                   Ydummy[0:(max_batches*batch_size),0:-1,0],
                   epochs=10,
                   batch_size=batch_size,
                   shuffle=False,
                   verbose=0)

test_arr = trainX.transpose((0,2,1))[50:50+batch_size:,:].astype('float32')
ErrorEnd = np.sum(Errors.eval({I:test_arr*F}))
print(ErrorStart, ErrorEnd)

math.sqrt(rnn.get_weights()[2][0]/(1-rnn.get_weights()[0][0][0]-rnn.get_weights()[1][0][0])*255)/F

kappa*F*F, rnn.get_weights()[2][0], rnn.get_weights()[0][0][0], rnn.get_weights()[1][0][0]



