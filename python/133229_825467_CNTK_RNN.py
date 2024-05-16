import numpy as np
import os
import sys
import cntk
from cntk.layers import Embedding, LSTM, GRU, Dense, Recurrence
from cntk import sequence
from common.params_lstm import *
from common.utils import *

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("CNTK: ", cntk.__version__)
print("GPU: ", get_gpu_name())

def create_symbol(CUDNN=True):
    # Weight initialiser from uniform distribution
    # Activation (unless states) is None
    with cntk.layers.default_options(init = cntk.glorot_uniform()):
        x = Embedding(EMBEDSIZE)(features) # output: list of len=BATCHSIZE of arrays with shape=(MAXLEN, EMBEDSIZE)
        
        # Since we have a vanilla RNN, instead of using the more flexible Recurrence(GRU) unit, which allows for
        # example LayerNormalisation to be added to the network, we can use optimized_rnnstack which quickly
        # goes down to the CuDNN level. This is another reason not to read much into the speed comparison because
        # it becomes a measure of which framework has the fastest way to go down to CuDNN.
        if not CUDNN:
            x = Recurrence(GRU(NUMHIDDEN))(x) # output: list of len=BATCHSIZE of arrays with shape=(MAXLEN, NUMHIDDEN)
        else:
            W = cntk.parameter((cntk.InferredDimension, 4))
            x = cntk.ops.optimized_rnnstack(x, W, NUMHIDDEN, num_layers=1, bidirectional=False, recurrent_op='gru')
        
        x = sequence.last(x) #o utput: array with shape=(BATCHSIZE, NUMHIDDEN)
        x = Dense(2)(x) # output: array with shape=(BATCHSIZE, 2)
        return x

def init_model(m):
    # Loss (dense labels); check if support for sparse labels
    loss = cntk.cross_entropy_with_softmax(m, labels)  
    # ADAM, set unit_gain to False to match others
    learner = cntk.adam(m.parameters,
                        lr=cntk.learning_rate_schedule(LR, cntk.UnitType.minibatch) ,
                        momentum=cntk.momentum_schedule(BETA_1), 
                        variance_momentum=cntk.momentum_schedule(BETA_2),
                        epsilon=EPS,
                        unit_gain=False)
    trainer = cntk.Trainer(m, (loss, cntk.classification_error(m, labels)), [learner])
    return trainer

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES, one_hot=True)# CNTK format\ny_train = y_train.astype(np.float32)\ny_test = y_test.astype(np.float32)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Placeholders\nfeatures = sequence.input_variable(shape=MAXFEATURES, is_sparse=True)\nlabels = cntk.input_variable(2)\n# Load symbol\nsym = create_symbol()')

get_ipython().run_cell_magic('time', '', 'trainer = init_model(sym)')

get_ipython().run_cell_magic('time', '', '# Train model\nfor j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        data_1hot = cntk.Value.one_hot(data, MAXFEATURES) #TODO: do this externally and generate batches of 1hot\n        trainer.train_minibatch({features: data_1hot, labels: label})\n    # Log (this is just last batch in epoch, not average of batches)\n    eval_error = trainer.previous_minibatch_evaluation_average\n    print("Epoch %d  |  Accuracy: %.6f" % (j+1, (1-eval_error)))')

get_ipython().run_cell_magic('time', '', '# Predict and then score accuracy\n# Apply softmax since that is only applied at training\n# with cross-entropy loss\nz = cntk.softmax(sym)\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = np.argmax(y_test[:n_samples], axis=-1)\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    data = cntk.Value.one_hot(data, MAXFEATURES)\n    predicted_label_probs = z.eval({features : data})\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = np.argmax(predicted_label_probs, axis=-1)\n    c += 1')

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

