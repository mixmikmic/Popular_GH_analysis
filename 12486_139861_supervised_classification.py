from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, MultivariateNormalTriL, Normal
from edward.util import rbf

ed.set_seed(42)

data = np.loadtxt('data/crabs_train.txt', delimiter=',')
data[data[:, 0] == -1, 0] = 0  # replace -1 label with 0 label

N = data.shape[0]  # number of data points
D = data.shape[1] - 1  # number of features

X_train = data[:, 1:]
y_train = data[:, 0]

print("Number of data points: {}".format(N))
print("Number of features: {}".format(D))

X = tf.placeholder(tf.float32, [N, D])
f = MultivariateNormalTriL(loc=tf.zeros(N), scale_tril=tf.cholesky(rbf(X)))
y = Bernoulli(logits=f)

qf = Normal(loc=tf.Variable(tf.random_normal([N])),
            scale=tf.nn.softplus(tf.Variable(tf.random_normal([N]))))

inference = ed.KLqp({f: qf}, data={X: X_train, y: y_train})
inference.run(n_iter=5000)

