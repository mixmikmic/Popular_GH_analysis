import os
import sys
import numpy as np
import mxnet as mx
from common.params_lstm import *
from common.utils import *

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("MXNet: ", mx.__version__)
print("GPU: ", get_gpu_name())

def create_symbol(CUDNN=True):
    # https://mxnet.incubator.apache.org/api/python/rnn.html
    data = mx.symbol.Variable('data')
    embedded_step = mx.symbol.Embedding(data=data, input_dim=MAXFEATURES, output_dim=EMBEDSIZE)
    
    # Fusing RNN layers across time step into one kernel
    # Improves speed but is less flexible
    # Currently only supported if using cuDNN on GPU
    if not CUDNN:
        gru_cell = mx.rnn.GRUCell(num_hidden=NUMHIDDEN)
    else:
        gru_cell = mx.rnn.FusedRNNCell(num_hidden=NUMHIDDEN, num_layers=1, mode='gru')
    
    begin_state = gru_cell.begin_state()
    # Call the cell to get the output of one time step for a batch.
    # TODO: TNC layout (sequence length, batch size, and feature dimensions) is faster for RNN
    outputs, states = gru_cell.unroll(length=MAXLEN, inputs=embedded_step, merge_outputs=False)
    
    fc1 = mx.symbol.FullyConnected(data=outputs[-1], num_hidden=2) 
    input_y = mx.symbol.Variable('softmax_label')  
    m = mx.symbol.SoftmaxOutput(data=fc1, label=input_y, name="softmax")
    return m

def init_model(m):
    if GPU:
        ctx = [mx.gpu(0)]
    else:
        ctx = mx.cpu()
    mod = mx.mod.Module(context=ctx, symbol=m)
    mod.bind(data_shapes=[('data', (BATCHSIZE, MAXLEN))],
             label_shapes=[('softmax_label', (BATCHSIZE, ))])
    # Glorot-uniform initializer
    mod.init_params(initializer=mx.init.Xavier(rnd_type='uniform'))
    mod.init_optimizer(optimizer='Adam', 
                       optimizer_params=(('learning_rate', LR),
                                         ('beta1', BETA_1),
                                         ('beta2', BETA_2),
                                         ('epsilon', EPS)))
    return mod

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES)\n\n# Use custom iterator instead of mx.io.NDArrayIter() for consistency\n# Wrap as DataBatch class\nwrapper_db = lambda args: mx.io.DataBatch(data=[mx.nd.array(args[0])], label=[mx.nd.array(args[1])])\n\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')

get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')

get_ipython().run_cell_magic('time', '', "# Train and log accuracy\nmetric = mx.metric.create('acc')\nfor j in range(EPOCHS):\n    #train_iter.reset()\n    metric.reset()\n    #for batch in train_iter:\n    for batch in map(wrapper_db, yield_mb(x_train, y_train, BATCHSIZE, shuffle=True)):\n        model.forward(batch, is_train=True) \n        model.update_metric(metric, batch.label)\n        model.backward()              \n        model.update()\n    print('Epoch %d, Training %s' % (j, metric.get()))")

get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(mx.io.NDArrayIter(x_test, batch_size=BATCHSIZE, shuffle=False))\ny_guess = np.argmax(y_guess.asnumpy(), axis=-1)')

print("Accuracy: ", sum(y_guess == y_test)/len(y_guess))

