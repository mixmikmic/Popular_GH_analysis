import numpy as np
import os
import sys
import tensorflow as tf
from common.params_lstm import *
from common.utils import *

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print("GPU: ", get_gpu_name())

def create_symbol(CUDNN=True):
    word_vectors = tf.contrib.layers.embed_sequence(X, vocab_size=MAXFEATURES, embed_dim=EMBEDSIZE)
    word_list = tf.unstack(word_vectors, axis=1)
    
    if not CUDNN:
        cell = tf.contrib.rnn.GRUCell(NUMHIDDEN)
        outputs, states = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)
    else:
        # Using cuDNN since vanilla RNN
        cudnn_cell = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, 
                                                   num_units=NUMHIDDEN, 
                                                   input_size=EMBEDSIZE)
        params_size_t = cudnn_cell.params_size()
        params = tf.Variable(tf.random_uniform([params_size_t], -0.1, 0.1), validate_shape=False)   
        input_h = tf.Variable(tf.zeros([1, BATCHSIZE, NUMHIDDEN]))
        outputs, states = cudnn_cell(input_data=word_list,
                                     input_h=input_h,
                                     params=params)
        logits = tf.layers.dense(outputs[-1], 2, activation=None, name='output')
    return logits

def init_model(m):
    # Single-class labels, don't need dense one-hot
    # Expects unscaled logits, not output of tf.nn.softmax
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=m, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(LR, BETA_1, BETA_2, EPS)
    training_op = optimizer.minimize(loss)
    return training_op

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Place-holders\nX = tf.placeholder(tf.int32, shape=[None, MAXLEN])\ny = tf.placeholder(tf.int32, shape=[None])\nsym = create_symbol()')

get_ipython().run_cell_magic('time', '', 'model = init_model(sym)\ninit = tf.global_variables_initializer()\nsess = tf.Session()\nsess.run(init)')

get_ipython().run_cell_magic('time', '', '# Accuracy logging\ncorrect = tf.nn.in_top_k(sym, y, 1)\naccuracy = tf.reduce_mean(tf.cast(correct, tf.float32))\n\nfor j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        sess.run(model, feed_dict={X: data, y: label})\n    # Log\n    acc_train = sess.run(accuracy, feed_dict={X: data, y: label})\n    print(j, "Train accuracy:", acc_train)')

get_ipython().run_cell_magic('time', '', 'n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    pred = tf.argmax(sym, 1)\n    output = sess.run(pred, feed_dict={X: data})\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = output\n    c += 1')

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

