import numpy as np
import os
import sys
import theano.tensor as T
import theano
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as nl
import lasagne.objectives as obj
import lasagne.updates as upd
from common.params import *
from common.utils import *

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Theano: ", theano.__version__)
print("Lasagne: ", lasagne.__version__)
print("GPU: ", get_gpu_name())

#CuDNN auto-tune
theano.config.dnn.conv.algo_fwd = "time_once"
theano.config.dnn.conv.algo_bwd_filter = "time_once"
theano.config.dnn.conv.algo_bwd_data = "time_once"

def create_symbol():
    conv1 = L.Conv2DLayer(X, num_filters=50, filter_size=(3, 3), pad='same')
    conv2 = L.Conv2DLayer(conv1, num_filters=50, filter_size=(3, 3), pad='same')
    pool1 = L.MaxPool2DLayer(conv2, pool_size=(2, 2), stride=(2, 2))
    drop1 = L.DropoutLayer(pool1, 0.25)
    
    conv3 = L.Conv2DLayer(drop1, num_filters=100, filter_size=(3, 3), pad='same')
    conv4 = L.Conv2DLayer(conv3, num_filters=100, filter_size=(3, 3), pad='same')
    pool2 = L.MaxPool2DLayer(conv4, pool_size=(2, 2), stride=(2, 2))
    drop2 = L.DropoutLayer(pool2, 0.25)
    
    flatten = L.FlattenLayer(drop2)
    fc1 = L.DenseLayer(flatten, 512)
    drop4 = L.DropoutLayer(fc1, 0.5)
    pred = L.DenseLayer(drop4, N_CLASSES, name="output", nonlinearity=nl.softmax)
    
    return pred

def init_model(net):
    pred = L.get_output(net)
    params = L.get_all_params(net)
    xentropy = obj.categorical_crossentropy(pred, y)
    loss = T.mean(xentropy)
    # The tensorflow LR, MOMENTUM are slightly different
    updates = upd.momentum(loss, params, learning_rate=LR, momentum=MOMENTUM)
    return pred, loss, updates

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Place-holders\nX = L.InputLayer(shape=(None, 3, 32, 32))\ny = T.ivector("y")\n# Initialise model\nnet = create_symbol()')

get_ipython().run_cell_magic('time', '', 'pred, loss, updates = init_model(net)\n# Accuracy for logging\naccuracy = obj.categorical_accuracy(pred, y)\naccuracy = T.mean(T.cast(accuracy, theano.config.floatX))')

get_ipython().run_cell_magic('time', '', '# Compile functions\ntrain_func = theano.function([X.input_var, y], [loss, accuracy], updates=updates)\npred = L.get_output(net, deterministic=True)\npred_func = theano.function([X.input_var], T.argmax(pred, axis=1))')

get_ipython().run_cell_magic('time', '', 'for j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        loss, acc_train = train_func(data, label)\n    # Log\n    print(j, "Train accuracy:", acc_train)')

get_ipython().run_cell_magic('time', '', 'n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    output = pred_func(data)\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = output\n    c += 1')

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

