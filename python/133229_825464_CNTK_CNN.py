import numpy as np
import os
import sys
import cntk
from cntk.layers import Convolution2D, MaxPooling, Dense, Dropout
from common.params import *
from common.utils import *

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("CNTK: ", cntk.__version__)
print("GPU: ", get_gpu_name())

def create_symbol():
    # Weight initialiser from uniform distribution
    # Activation (unless states) is None
    with cntk.layers.default_options(init = cntk.glorot_uniform(), activation = cntk.relu):
        x = Convolution2D(filter_shape=(3, 3), num_filters=50, pad=True)(features)
        x = Convolution2D(filter_shape=(3, 3), num_filters=50, pad=True)(x)
        x = MaxPooling((2, 2), strides=(2, 2), pad=False)(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(filter_shape=(3, 3), num_filters=100, pad=True)(x)
        x = Convolution2D(filter_shape=(3, 3), num_filters=100, pad=True)(x)
        x = MaxPooling((2, 2), strides=(2, 2), pad=False)(x)
        x = Dropout(0.25)(x)    
        
        x = Dense(512)(x)
        x = Dropout(0.5)(x)
        x = Dense(N_CLASSES, activation=None)(x)
        return x

def init_model(m):
    # Loss (dense labels); check if support for sparse labels
    loss = cntk.cross_entropy_with_softmax(m, labels)  
    # Momentum SGD
    # https://github.com/Microsoft/CNTK/blob/master/Manual/Manual_How_to_use_learners.ipynb
    # unit_gain=False: momentum_direction = momentum*old_momentum_direction + gradient
    # if unit_gain=True then ...(1-momentum)*gradient
    learner = cntk.momentum_sgd(m.parameters,
                                lr=cntk.learning_rate_schedule(LR, cntk.UnitType.minibatch) ,
                                momentum=cntk.momentum_schedule(MOMENTUM), 
                                unit_gain=False)
    trainer = cntk.Trainer(m, (loss, cntk.classification_error(m, labels)), [learner])
    return trainer

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True, one_hot=True)\n# CNTK format\ny_train = y_train.astype(np.float32)\ny_test = y_test.astype(np.float32)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Placeholders\nfeatures = cntk.input_variable((3, 32, 32), np.float32)\nlabels = cntk.input_variable(N_CLASSES, np.float32)\n# Load symbol\nsym = create_symbol()')

get_ipython().run_cell_magic('time', '', 'trainer = init_model(sym)')

get_ipython().run_cell_magic('time', '', '# Train model\nfor j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        trainer.train_minibatch({features: data, labels: label})\n    # Log (this is just last batch in epoch, not average of batches)\n    eval_error = trainer.previous_minibatch_evaluation_average\n    print("Epoch %d  |  Accuracy: %.6f" % (j+1, (1-eval_error)))')

get_ipython().run_cell_magic('time', '', '# Predict and then score accuracy\n# Apply softmax since that is only applied at training\n# with cross-entropy loss\nz = cntk.softmax(sym)\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = np.argmax(y_test[:n_samples], axis=-1)\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    predicted_label_probs = z.eval({features : data})\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = np.argmax(predicted_label_probs, axis=-1)\n    c += 1')

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

