import numpy as np
import os
import sys
import tensorflow as tf
from common.params import *
from common.utils import *

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = "1"

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print("GPU: ", get_gpu_name())

def create_symbol(training):
    """ TF pooling requires a boolean flag for dropout, faster when using
    'channels_first' for data_format """
    conv1 = tf.layers.conv2d(X, filters=50, kernel_size=(3, 3), 
                             padding='same', data_format='channels_first')
    relu1 = tf.nn.relu(conv1)
    conv2 = tf.layers.conv2d(relu1, filters=50, kernel_size=(3, 3), 
                             padding='same', data_format='channels_first')
    pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), 
                                    padding='valid', data_format='channels_first')
    relu2 = tf.nn.relu(pool1)
    drop1 = tf.layers.dropout(relu2, 0.25, training=training)
    
    conv3 = tf.layers.conv2d(drop1, filters=100, kernel_size=(3, 3), 
                             padding='same', data_format='channels_first')
    relu3 = tf.nn.relu(conv3)
    conv4 = tf.layers.conv2d(relu3, filters=100, kernel_size=(3, 3), 
                             padding='same', data_format='channels_first')
    pool2 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(2, 2), 
                                    padding='valid', data_format='channels_first')
    relu4 = tf.nn.relu(pool2)
    drop2 = tf.layers.dropout(relu4, 0.25, training=training)   
    
    flatten = tf.reshape(drop2, shape=[-1, 100*8*8])
    fc1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
    drop3 = tf.layers.dropout(fc1, 0.5, training=training)
    logits = tf.layers.dense(drop3, N_CLASSES, name='output')
    return logits

def init_model(m):
    # Single-class labels, don't need dense one-hot
    # Expects unscaled logits, not output of tf.nn.softmax
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=m, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=MOMENTUM)
    training_op = optimizer.minimize(loss)
    return training_op

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Place-holders\nX = tf.placeholder(tf.float32, shape=[None, 3, 32, 32])\ny = tf.placeholder(tf.int32, shape=[None])\ntraining = tf.placeholder(tf.bool)  # Indicator for dropout layer\n# Initialise model\nsym = create_symbol(training)')

get_ipython().run_cell_magic('time', '', 'model = init_model(sym)\ninit = tf.global_variables_initializer()\nsess = tf.Session()\nsess.run(init)\n# Accuracy logging\ncorrect = tf.nn.in_top_k(sym, y, 1)\naccuracy = tf.reduce_mean(tf.cast(correct, tf.float32))')

get_ipython().run_cell_magic('time', '', 'for j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        sess.run(model, feed_dict={X: data, y: label, training: True})\n    # Log\n    acc_train = sess.run(accuracy, feed_dict={X: data, y: label, training: True})\n    print(j, "Train accuracy:", acc_train)')

get_ipython().run_cell_magic('time', '', 'n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    pred = tf.argmax(sym,1)\n    output = sess.run(pred, feed_dict={X: data, training: False})\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = output\n    c += 1')

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

