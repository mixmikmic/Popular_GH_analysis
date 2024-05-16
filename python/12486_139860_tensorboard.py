get_ipython().magic('matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

plt.style.use('ggplot')

def build_toy_dataset(N, w):
  D = len(w)
  x = np.random.normal(0.0, 2.0, size=(N, D))
  y = np.dot(x, w) + np.random.normal(0.0, 0.01, size=N)
  return x, y

ed.set_seed(42)

N = 40  # number of data points
D = 5  # number of features

w_true = np.random.randn(D) * 0.5
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)

with tf.name_scope('model'): 
  X = tf.placeholder(tf.float32, [N, D], name="X")
  w = Normal(loc=tf.zeros(D, name="weights/loc"), scale=tf.ones(D, name="weights/loc"), name="weights")
  b = Normal(loc=tf.zeros(1, name="bias/loc"), scale=tf.ones(1, name="bias/scale"), name="bias")
  y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N, name="y/scale"), name="y")

with tf.name_scope("posterior"):
  qw = Normal(loc=tf.Variable(tf.random_normal([D]), name="qw/loc"),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]), name="qw/unconstrained_scale")), 
              name="qw")
  qb = Normal(loc=tf.Variable(tf.random_normal([1]), name="qb/loc"),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]), name="qb/unconstrained_scale")), 
              name="qb")

inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run(n_samples=5, n_iter=250, logdir='log/n_samples_5')

