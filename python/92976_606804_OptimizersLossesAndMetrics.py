from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(64, input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

import tensorflow as tf

# Use a tensorflow optimizer
pgd = optimizers.TFOptimizer(tf.train.ProximalGradientDescentOptimizer(0.01))

model.compile(loss='mean_squared_error', optimizer='sgd')

from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')

model.compile(loss=tf.nn.log_poisson_loss, optimizer='sgd')

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])

from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])



