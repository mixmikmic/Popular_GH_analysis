get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow,numpy,h5py")

# Note that executing the following code 
# cell will download the MNIST dataset
# and save all the 60,000 images as separate JPEG
# files. This might take a few minutes depending
# on your machine.

import numpy as np
from helper import mnist_export_to_jpg

np.random.seed(123)
mnist_export_to_jpg(path='./')

import os

for i in ('train', 'valid', 'test'):
    print('mnist_%s subdirectories' % i, os.listdir('mnist_%s' % i))

get_ipython().magic('matplotlib inline')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

some_img = os.path.join('./mnist_train/9/', os.listdir('./mnist_train/9/')[0])

img = mpimg.imread(some_img)
print(img.shape)
plt.imshow(img, cmap='binary');

import numpy as np
import h5py
import glob


def images_to_h5(data_stempath='./mnist_',
                 width=28, height=28, channels=1,
                 shuffle=False, random_seed=None):
    
    with h5py.File('mnist_batches.h5', 'w') as h5f:
    
        for s in ['train', 'valid', 'test']:
            img_paths = [p for p in glob.iglob('%s%s/**/*.jpg' % 
                                       (data_stempath, s), 
                                        recursive=True)]

            dset1 = h5f.create_dataset('%s/images' % s, 
                                       shape=[len(img_paths), 
                                              width, height, channels], 
                                       compression=None,
                                       dtype='uint8')
            dset2 = h5f.create_dataset('%s/labels' % s, 
                                       shape=[len(img_paths)], 
                                       compression=None,
                                       dtype='uint8')
            dset3 = h5f.create_dataset('%s/file_ids' % s, 
                                       shape=[len(img_paths)], 
                                       compression=None,
                                       dtype='S5')
            
            rand_indices = np.arange(len(img_paths))
            
            if shuffle:
                rng = np.random.RandomState(random_seed)
                rng.shuffle(rand_indices)

            for idx, path in enumerate(img_paths):

                rand_idx = rand_indices[idx]
                label = int(os.path.basename(os.path.dirname(path)))
                image = mpimg.imread(path)
                dset1[rand_idx] = image.reshape(width, height, channels)
                dset2[rand_idx] = label
                dset3[rand_idx] = np.array([os.path.basename(path)], dtype='S6')

images_to_h5(shuffle=True, random_seed=123)

with h5py.File('mnist_batches.h5', 'r') as h5f:
    print(h5f['train/images'].shape)
    print(h5f['train/labels'].shape)
    print(h5f['train/file_ids'].shape)

with h5py.File('mnist_batches.h5', 'r') as h5f:

    plt.imshow(h5f['train/images'][0][:, :, -1], cmap='binary');
    print('Class label:', h5f['train/labels'][0])
    print('File ID:', h5f['train/file_ids'][0])

class BatchLoader():
    def __init__(self, minibatches_path, 
                 normalize=True):
        
        self.minibatches_path = minibatches_path
        self.normalize = normalize
        self.num_train = 45000
        self.num_valid = 5000
        self.num_test = 10000
        self.n_classes = 10


    def load_train_epoch(self, batch_size=50, onehot=False,
                         shuffle_batch=False, prefetch_batches=1, seed=None):
        for batch_x, batch_y in self._load_epoch(which='train',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_batch=shuffle_batch,
                                                 prefetch_batches=prefetch_batches, 
                                                 seed=seed):
            yield batch_x, batch_y

    def load_test_epoch(self, batch_size=50, onehot=False,
                        shuffle_batch=False, prefetch_batches=1, seed=None):
        for batch_x, batch_y in self._load_epoch(which='test',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_batch=shuffle_batch,
                                                 prefetch_batches=prefetch_batches,
                                                 seed=seed):
            yield batch_x, batch_y
            
    def load_validation_epoch(self, batch_size=50, onehot=False,
                              shuffle_batch=False, prefetch_batches=1, seed=None):
        for batch_x, batch_y in self._load_epoch(which='valid',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_batch=shuffle_batch,
                                                 prefetch_batches=prefetch_batches, 
                                                 seed=seed):
            yield batch_x, batch_y

    def _load_epoch(self, which='train', batch_size=50, onehot=False,
                    shuffle_batch=False, prefetch_batches=1, seed=None):
        
        prefetch_size = prefetch_batches * batch_size
        
        if shuffle_batch:
            rgen = np.random.RandomState(seed)

        with h5py.File(self.minibatches_path, 'r') as h5f:
            indices = np.arange(h5f['%s/images' % which].shape[0])
            
            for start_idx in range(0, indices.shape[0] - prefetch_size + 1,
                                   prefetch_size):           
            

                x_batch = h5f['%s/images' % which][start_idx:start_idx + prefetch_size]
                x_batch = x_batch.astype(np.float32)
                y_batch = h5f['%s/labels' % which][start_idx:start_idx + prefetch_size]

                if onehot:
                    y_batch = (np.arange(self.n_classes) == 
                               y_batch[:, None]).astype(np.uint8)

                if self.normalize:
                    # normalize to [0, 1] range
                    x_batch = x_batch.astype(np.float32) / 255.

                if shuffle_batch:
                    rand_indices = np.arange(prefetch_size)
                    rgen.shuffle(rand_indices)
                    x_batch = x_batch[rand_indices]
                    y_batch = y_batch[rand_indices]

                for batch_idx in range(0, x_batch.shape[0] - batch_size + 1,
                                       batch_size):
                    
                    yield (x_batch[batch_idx:batch_idx + batch_size], 
                           y_batch[batch_idx:batch_idx + batch_size])

batch_loader = BatchLoader(minibatches_path='./mnist_batches.h5', 
                           normalize=True)

for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=50, onehot=True):
    print(batch_x.shape)
    print(batch_y.shape)
    break

cnt = 0
for batch_x, batch_y in batch_loader.load_train_epoch(
        batch_size=100, onehot=True):
    cnt += batch_x.shape[0]
    
print('One training epoch contains %d images' % cnt)

def one_epoch():
    for batch_x, batch_y in batch_loader.load_train_epoch(
            batch_size=100, onehot=True):
        pass
    
get_ipython().magic('timeit one_epoch()')

def one_epoch():
    for batch_x, batch_y in batch_loader.load_train_epoch(
            batch_size=100, shuffle_batch=True, prefetch_batches=4, 
            seed=123, onehot=True):
        pass
    
get_ipython().magic('timeit one_epoch()')

import tensorflow as tf

##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
training_epochs = 15
batch_size = 100

# Architecture
n_hidden_1 = 128
n_hidden_2 = 256
height, width = 28, 28
n_classes = 10


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    
    tf.set_random_seed(123)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, height, width, 1], name='features')
    tf_x_flat = tf.reshape(tf_x, shape=[-1, height*width])
    tf_y = tf.placeholder(tf.int32, [None, n_classes], name='targets')

    # Model parameters
    weights = {
        'h1': tf.Variable(tf.truncated_normal([width*height, n_hidden_1], stddev=0.1)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=0.1))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    # Multilayer perceptron
    layer_1 = tf.add(tf.matmul(tf_x_flat, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(out_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

##########################
### TRAINING & EVALUATION
##########################

batch_loader = BatchLoader(minibatches_path='./mnist_batches.h5', 
                           normalize=True)

# preload small validation set
# by unpacking the generator
[valid_data] = batch_loader.load_validation_epoch(batch_size=5000, 
                                                   onehot=True)
valid_x, valid_y = valid_data[0], valid_data[1]
del valid_data

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.

        n_batches = 0
        for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=batch_size, 
                                                              onehot=True,
                                                              shuffle_batch=True,
                                                              prefetch_batches=10,
                                                              seed=epoch):
            n_batches += 1
            _, c = sess.run(['train', 'cost:0'], feed_dict={'features:0': batch_x,
                                                            'targets:0': batch_y.astype(np.int)})
            avg_cost += c
        
        train_acc = sess.run('accuracy:0', feed_dict={'features:0': batch_x,
                                                      'targets:0': batch_y})
        
        valid_acc = sess.run('accuracy:0', feed_dict={'features:0': valid_x,
                                                      'targets:0': valid_y})  
        
        print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / n_batches), end="")
        print(" | MbTrain/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
        
        
    # imagine test set is too large to fit into memory:
    test_acc, cnt = 0., 0
    for test_x, test_y in batch_loader.load_test_epoch(batch_size=100, 
                                                       onehot=True):   
        cnt += 1
        acc = sess.run(accuracy, feed_dict={'features:0': test_x,
                                            'targets:0': test_y})
        test_acc += acc
    print('Test ACC: %.3f' % (test_acc / cnt))

