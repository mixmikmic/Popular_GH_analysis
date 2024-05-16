import numpy as np
import tensorflow as tf
from mlp import BO
import os 

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

optimizer = ["sgd", "momentum", "nestrov_momentum", "adagrad", "adadelta", "rmsprop", "adam"]
learning_rate = [0.0001, 0.001, 0.01, 0.1]

x_train = mnist.train.images
y_train = mnist.train.labels
x_valid = mnist.validation.images
y_valid = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels
print (x_test.shape)
print (y_test.shape)

model = BO(x_train, y_train, x_valid, y_valid, x_test, y_test)
print ("[Model Initialized]")

model.build_graph()

model.compile_graph(optimize = optimizer[1], learning_rate = learning_rate[0])

model.train(summary_dir = "/tmp/mnist/"+optimizer[1]+"_"+str(learning_rate[0]))

