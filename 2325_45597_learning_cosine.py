import os
os.environ['THEANO_FLAGS']='mode=FAST_COMPILE,optimizer=None,device=cpu,floatX=float32'

import numpy as np
import sklearn.cross_validation as skcv
#x = np.linspace(0, 5*np.pi, num=10000, dtype=np.float32)
x = np.linspace(0, 4*np.pi, num=10000, dtype=np.float32)
y = np.cos(x)

train, test = skcv.train_test_split(np.arange(x.shape[0]))
print train.shape
print test.shape

import pylab as pl
get_ipython().magic('matplotlib inline')
pl.plot(x, y)

X_train = x[train].reshape(-1, 1)
y_train = y[train]

print "x_train : ", X_train.min(), X_train.max()
print X_train.shape
print "y_train : ", y_train.min(), y_train.max()
print y_train.shape
assert X_train.dtype == np.float32
assert y_train.dtype == np.float32

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(1, 4, init='lecun_uniform'))
model.add(Activation('tanh'))
model.add(Dense(4, 1, init='lecun_uniform'))
model.add(Activation('tanh'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

print model.get_weights()
history = model.fit(scaler.transform(X_train), y_train, nb_epoch=10, batch_size=64, shuffle=True)

y_pred = model.predict(scaler.transform(x.reshape(-1, 1)))

model.get_weights()

pl.plot(x, y_pred, c='r', label='y_pred')
pl.plot(x, y, c='b', label='y')
pl.legend()

def train_plot_prediction(n_hidden):
    model = Sequential()
    model.add(Dense(1, n_hidden, init='lecun_uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(n_hidden, 1, init='lecun_uniform'))
    model.add(Activation('tanh'))
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    
    history = model.fit(scaler.transform(X_train), y_train, nb_epoch=5, batch_size=64, shuffle=True,
                       verbose=False)
    
    y_pred = model.predict(scaler.transform(x.reshape(-1, 1)))
    
    pl.figure(figsize=(10, 4))
    pl.subplot(211)
    pl.title('train loss')
    pl.plot(history.epoch, history.history['loss'], label='loss')
    pl.subplot(212)
    pl.title('prediction vs ground truth')
    pl.plot(x, y_pred, c='r', label='y_pred')
    pl.plot(x, y, c='b', label='y')
    pl.legend()
    pl.tight_layout()

train_plot_prediction(1)

train_plot_prediction(2)

train_plot_prediction(3)

train_plot_prediction(4)

train_plot_prediction(5)

train_plot_prediction(10)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, max_depth=10).fit(scaler.transform(X_train), y_train)

y_pred_rf = rf.predict(scaler.transform(x.reshape(-1, 1)))

pl.figure(figsize=(10, 4))
pl.plot(x, y_pred_rf, c='r', label='y_pred')
pl.plot(x, y, c='b', label='y')
pl.legend()



