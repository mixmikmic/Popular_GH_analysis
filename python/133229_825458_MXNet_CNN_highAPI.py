import os
import sys
import numpy as np
import mxnet as mx
from common.params import *
from common.utils import *

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("MXNet: ", mx.__version__)
print("GPU: ", get_gpu_name())

def create_symbol():
    data = mx.symbol.Variable('data')
    # size = [(old-size - kernel + 2*padding)/stride]+1
    # if kernel = 3, pad with 1 either side
    conv1 = mx.symbol.Convolution(data=data, num_filter=50, pad=(1,1), kernel=(3,3))
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    conv2 = mx.symbol.Convolution(data=relu1, num_filter=50, pad=(1,1), kernel=(3,3))
    pool1 = mx.symbol.Pooling(data=conv2, pool_type="max", kernel=(2,2), stride=(2,2))
    relu2 = mx.symbol.Activation(data=pool1, act_type="relu")
    drop1 = mx.symbol.Dropout(data=relu2, p=0.25)
    
    conv3 = mx.symbol.Convolution(data=drop1, num_filter=100, pad=(1,1), kernel=(3,3))
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(data=relu3, num_filter=100, pad=(1,1), kernel=(3,3))
    pool2 = mx.symbol.Pooling(data=conv4, pool_type="max", kernel=(2,2), stride=(2,2))
    relu4 = mx.symbol.Activation(data=pool2, act_type="relu")
    drop2 = mx.symbol.Dropout(data=relu4, p=0.25)
           
    flat1 = mx.symbol.Flatten(data=drop2)
    fc1 = mx.symbol.FullyConnected(data=flat1, num_hidden=512)
    relu7 = mx.symbol.Activation(data=fc1, act_type="relu")
    drop4 = mx.symbol.Dropout(data=relu7, p=0.5)
    fc2 = mx.symbol.FullyConnected(data=drop4, num_hidden=N_CLASSES) 
    
    input_y = mx.symbol.Variable('softmax_label')  
    m = mx.symbol.SoftmaxOutput(data=fc2, label=input_y, name="softmax")
    return m

def init_model(m):
    if GPU:
        ctx = [mx.gpu(0)]
    else:
        ctx = mx.cpu()
    mod = mx.mod.Module(context=ctx, symbol=m)
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    # Glorot-uniform initializer
    mod.init_params(initializer=mx.init.Xavier(rnd_type='uniform'))
    mod.init_optimizer(optimizer='sgd', 
                       optimizer_params=(('learning_rate', LR), ('momentum', MOMENTUM), ))
    return mod

get_ipython().run_cell_magic('time', '', '# Data into format for library\n#x_train, x_test, y_train, y_test = mnist_for_library(channel_first=True)\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\n# Load data-iterator\ntrain_iter = mx.io.NDArrayIter(x_train, y_train, BATCHSIZE, shuffle=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')

get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')

get_ipython().run_cell_magic('time', '', "# Train and log accuracy\nmetric = mx.metric.create('acc')\nfor j in range(EPOCHS):\n    train_iter.reset()\n    metric.reset()\n    for batch in train_iter:\n        model.forward(batch, is_train=True) \n        model.update_metric(metric, batch.label)\n        model.backward()              \n        model.update()\n    print('Epoch %d, Training %s' % (j, metric.get()))")

get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(mx.io.NDArrayIter(x_test, batch_size=BATCHSIZE, shuffle=False))\ny_guess = np.argmax(y_guess.asnumpy(), axis=-1)')

print("Accuracy: ", sum(y_guess == y_test)/len(y_guess))

