import os

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from models.models_1_3 import MNIST_DNN, MNIST_G, MNIST_D
from utils import plot

get_ipython().magic('matplotlib inline')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

logdir = './tf_logs/1_3_AM_Code/'
ckptdir = logdir + 'model'

if not os.path.exists(logdir):
    os.mkdir(logdir)

with tf.name_scope('Classifier'):

    # Initialize neural network
    DNN = MNIST_DNN('DNN')

    # Setup training process
    lmda = tf.placeholder_with_default(0.01, shape=[], name='lambda')
    X = tf.placeholder(tf.float32, [None, 784], name='X')
    Y = tf.placeholder(tf.float32, [None, 10], name='Y')

    tf.add_to_collection('placeholders', lmda)
    tf.add_to_collection('placeholders', X)
    tf.add_to_collection('placeholders', Y)

    code, logits = DNN(X)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

    optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=DNN.vars)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_summary = tf.summary.scalar('Cost', cost)
accuray_summary = tf.summary.scalar('Accuracy', accuracy)
summary = tf.summary.merge_all()

with tf.name_scope('GAN'):

    G = MNIST_G(z_dim=100, name='Generator')
    D = MNIST_D(name='Discriminator')

    X_fake = G(code)
    D_real = D(X)
    D_fake = D(X_fake, reuse=True)
    code_fake, logits_fake = DNN(X_fake, reuse=True)

    D_cost = -tf.reduce_mean(tf.log(D_real + 1e-7) + tf.log(1 - D_fake + 1e-7))
    G_cost = -tf.reduce_mean(tf.log(D_fake + 1e-7)) + tf.nn.l2_loss(X_fake - X) + tf.nn.l2_loss(code_fake - code)

    D_optimizer = tf.train.AdamOptimizer().minimize(D_cost, var_list=D.vars)
    G_optimizer = tf.train.AdamOptimizer().minimize(G_cost, var_list=G.vars)

with tf.name_scope('Prototype'):
    
    code_mean = tf.placeholder(tf.float32, [10, 100], name='code_mean')
    code_prototype = tf.get_variable('code_prototype', shape=[10, 100], initializer=tf.random_normal_initializer())

    X_prototype = G(code_prototype, reuse=True)
    Y_prototype = tf.one_hot(tf.cast(tf.lin_space(0., 9., 10), tf.int32), depth=10)
    _, logits_prototype = DNN(X_prototype, reuse=True)

    # Objective function definition
    cost_prototype = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_prototype, labels=Y_prototype))                      + lmda * tf.nn.l2_loss(code_prototype - code_mean)

    optimizer_prototype = tf.train.AdamOptimizer().minimize(cost_prototype, var_list=[code_prototype])

# Add the subgraph nodes to a collection so that they can be used after training of the network
tf.add_to_collection('prototype', code)
tf.add_to_collection('prototype', code_mean)
tf.add_to_collection('prototype', code_prototype)
tf.add_to_collection('prototype', X_prototype)
tf.add_to_collection('prototype', Y_prototype)
tf.add_to_collection('prototype', logits_prototype)
tf.add_to_collection('prototype', cost_prototype)
tf.add_to_collection('prototype', optimizer_prototype)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Hyper parameters
training_epochs = 15
batch_size = 100

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_cost = 0
    avg_acc = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c, a, summary_str = sess.run([optimizer, cost, accuracy, summary], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch
        avg_acc += a / total_batch
        
        file_writer.add_summary(summary_str, epoch * total_batch + i)
    
    print('Epoch: {:04d} cost = {:.9f} accuracy = {:.9f}'.format(epoch + 1, avg_cost, avg_acc))
    
    saver.save(sess, ckptdir)

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# Hyper parameters
training_epochs = 25
batch_size = 100
img_epoch = 1

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples / batch_size)
    avg_D_cost = 0
    avg_G_cost = 0
    
    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs}

        _, D_c = sess.run([D_optimizer, D_cost], feed_dict=feed_dict)
        _, G_c = sess.run([G_optimizer, G_cost], feed_dict=feed_dict)

        avg_D_cost += D_c / total_batch
        avg_G_cost += G_c / total_batch
        
    print('Epoch: {:04d} G cost = {:.9f} D cost = {:.9f}'.format(epoch + 1, avg_G_cost, avg_D_cost))

# Uncomment this code if you want to see the generated images.
#
#     if (epoch + 1) % img_epoch == 0:
#         samples = sess.run(X_fake, feed_dict={X: mnist.test.images[:16, :]})
#         fig = plot(samples, 784, 1)
#         plt.savefig('./assets/1_3_AM_Code/G_{:04d}.png'.format(epoch), bbox_inches='tight')
#         plt.close(fig)
    
    saver.save(sess, ckptdir)

sess.close()

tf.reset_default_graph()

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

# Get necessary placeholders
placeholders = tf.get_collection('placeholders')
lmda = placeholders[0]
X = placeholders[1]

# Get prototype nodes
prototype = tf.get_collection('prototype')
code = prototype[0]
code_mean = prototype[1]
X_prototype = prototype[3]
cost_prototype = prototype[6]
optimizer_prototype = prototype[7]

images = mnist.train.images
labels = mnist.train.labels

code_means = []
for i in range(10):
    imgs = images[np.argmax(labels, axis=1) == i]
    img_codes = sess.run(code, feed_dict={X: imgs})
    code_means.append(np.mean(img_codes, axis=0))

for epoch in range(15000):
    _, c = sess.run([optimizer_prototype, cost_prototype], feed_dict={lmda: 0.1, code_mean: code_means})
    
    if epoch % 500 == 0:
        print('Epoch: {:05d} Cost = {:.9f}'.format(epoch, c))
    
X_prototypes = sess.run(X_prototype)

sess.close()

plt.figure(figsize=(15,15))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    plt.imshow(np.reshape(X_prototypes[2 * i], [28, 28]), cmap='gray', interpolation='none')
    plt.title('Digit: {}'.format(2 * i))
    plt.colorbar()
    
    plt.subplot(5, 2, 2 * i + 2)
    plt.imshow(np.reshape(X_prototypes[2 * i + 1], [28, 28]), cmap='gray', interpolation='none')
    plt.title('Digit: {}'.format(2 * i + 1))
    plt.colorbar()

plt.tight_layout()



