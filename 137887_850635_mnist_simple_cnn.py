### Imports
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

### 1. Get the dataset
mnist = input_data.read_data_sets("mnist_dataset/", one_hot=True, reshape=False)

### 2. Pre-process the dataset

### 3. Define the hyper params
learning_rate = 5e-3
num_steps = 500
batch_size = 128
test_valid_size = 256

### 4. Define the architecture of your CNN
n_labels = 10

images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
labels = tf.placeholder(tf.float32, shape=[None, n_labels])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.truncated_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.truncated_normal([1024, n_labels]))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32])),
    'bc2': tf.Variable(tf.truncated_normal([64])),
    'bd1': tf.Variable(tf.truncated_normal([1024])),
    'out': tf.Variable(tf.truncated_normal([n_labels]))
}

def conv2d(input_vol, W, b, stride=1):
    conv_layer = tf.nn.conv2d(input_vol, W, strides=[1, stride, stride, 1], padding='SAME')
    conv_layer = tf.nn.bias_add(conv_layer, b)
    return tf.nn.relu(conv_layer)

def maxpool2d(input_vol, k=2, stride=2):
    return tf.nn.max_pool(input_vol, ksize=[1, k, k, 1],
                         strides=[1, stride, stride, 1],
                         padding='SAME')

def convnet(image, weights, biases, keep_prob):
    # CONV1 [28x28x1] * 32[5x5x1] -> [28x28x32]
    conv1 = conv2d(images, weights['wc1'], biases['bc1'])
    # POOL1 [28x28x32] -> [14x14x32]
    conv1 = maxpool2d(conv1)
    
    # CONV2 [14x14x32] * [5x5x64] -> [14x14x64]
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # POOL2 [14x14x64] -> [7x7x64]
    conv2 = maxpool2d(conv2)
    
    # FC1 [7x7x64] -> [1024]
    fc1 = tf.reshape(conv2, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.matmul(fc1, weights['wd1']) + biases['bd1']
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
    
    # OUT
    out = tf.matmul(fc1, weights['out']) + biases['out']
    return out

### 5. Run the training session and track the loss & validation accuracy
logits = convnet(images, weights, biases, keep_prob)

# Define the loss and the optimisation function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Accuracy
correct_predictions = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Run the graph
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    
    for step_i in range(num_steps):
        batch = mnist.train.next_batch(batch_size)
        session.run(optimiser,
                   feed_dict={
                       images: batch[0],
                       labels: batch[1],
                       keep_prob: 0.75
                   })

        # Calculate the batch loss and validation accuracy
        if step_i % 10 == 0: 
            loss = session.run(cost, feed_dict={
                images: batch[0],
                labels: batch[1],
                keep_prob: 1.
            })

            acc = session.run(accuracy, feed_dict={
                images: mnist.validation.images[:test_valid_size],
                labels: mnist.validation.labels[:test_valid_size],
                keep_prob: 1.
            })

            print('Step: {:>2} '
                  'Loss: {:>10.4f} Val Acc {:.6f}'.format(
                step_i, loss, acc))
            
    # 6. Calculate the test accuracy
    test_acc = session.run(accuracy, feed_dict={
        images: mnist.test.images[:test_valid_size],
        labels: mnist.test.labels[:test_valid_size],
        keep_prob: 1.
    })
    print('Testing Accuracy: {:.4f}'.format(test_acc))



