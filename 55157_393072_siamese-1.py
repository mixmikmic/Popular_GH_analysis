get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow")

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


##########################
### SETTINGS
##########################

# General settings

random_seed = 0

# Hyperparameters
learning_rate = 0.001
training_epochs = 5
batch_size = 100
margin = 1.0

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 1 # for 'true' and 'false' matches


def fully_connected(inputs, output_nodes, activation=None, seed=None):

    input_nodes = inputs.get_shape().as_list()[1]
    weights = tf.get_variable(name='weights', 
                              shape=(input_nodes, output_nodes),
                              initializer=tf.truncated_normal_initializer(
                                  mean=0.0,
                                  stddev=0.001,
                                  dtype=tf.float32,
                                  seed=seed))

    biases = tf.get_variable(name='biases', 
                             shape=(output_nodes,),
                             initializer=tf.constant_initializer(
                                 value=0.0, 
                                 dtype=tf.float32))
                              
    act = tf.matmul(inputs, weights) + biases
    if activation is not None:
        act = activation(act)
    return act


def euclidean_distance(x_1, x_2):
    return tf.sqrt(tf.maximum(tf.sum(
        tf.square(x - y), axis=1, keepdims=True), 1e-06))

def contrastive_loss(x_1, x_2, margin=1.0):
    return (x_1 * tf.square(x_2) +
            (1.0 - x_1) * tf.square(tf.maximum(margin - x_2, 0.)))


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(random_seed)

    # Input data
    tf_x_1 = tf.placeholder(tf.float32, [None, n_input], name='inputs_1')
    tf_x_2 = tf.placeholder(tf.float32, [None, n_input], name='inputs_2')
    tf_y = tf.placeholder(tf.float32, [None], 
                          name='targets') # here: 'true' or 'false' valuess

    # Siamese Network
    def build_mlp(inputs):
        with tf.variable_scope('fc_1'):
            layer_1 = fully_connected(inputs, n_hidden_1, 
                                      activation=tf.nn.relu)
        with tf.variable_scope('fc_2'):
            layer_2 = fully_connected(layer_1, n_hidden_2, 
                                      activation=tf.nn.relu)
        with tf.variable_scope('fc_3'):
            out_layer = fully_connected(layer_2, n_classes, 
                                        activation=tf.nn.relu)

        return out_layer
    
    
    with tf.variable_scope('siamese_net', reuse=False):
        pred_left = build_mlp(tf_x_1)
    with tf.variable_scope('siamese_net', reuse=True):
        pred_right = build_mlp(tf_x_2)
    
    # Loss and optimizer
    loss = contrastive_loss(pred_left, pred_right)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')
    
##########################
### TRAINING & EVALUATION
##########################

np.random.seed(random_seed) # set seed for mnist shuffling
mnist = input_data.read_data_sets("./", one_hot=False)

with tf.Session(graph=g) as sess:
    
    print('Initializing variables:')
    sess.run(tf.global_variables_initializer())
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope='siamese_net'):
        print(i)

    for epoch in range(training_epochs):
        avg_cost = 0.
        
        total_batch = mnist.train.num_examples // batch_size // 2

        for i in range(total_batch):
            
            batch_x_1, batch_y_1 = mnist.train.next_batch(batch_size)
            batch_x_2, batch_y_2 = mnist.train.next_batch(batch_size)
            batch_y = (batch_y_1 == batch_y_2).astype('float32')
            
            _, c = sess.run(['train', 'cost:0'], feed_dict={'inputs_1:0': batch_x_1,
                                                            'inputs_2:0': batch_x_2,
                                                            'targets:0': batch_y})
            avg_cost += c

        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)))

