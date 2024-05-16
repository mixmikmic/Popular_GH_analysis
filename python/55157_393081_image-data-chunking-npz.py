get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebastian Raschka' -v -p tensorflow,numpy")

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
import glob


def images_to_pickles(data_stempath='./mnist_', which_set='train', 
                      archive_size=5000, width=28, height=28, channels=1,
                      shuffle=False, seed=None):
    
    if not os.path.exists('%snpz' % data_stempath):
        os.mkdir('%snpz' % data_stempath)
        
    img_paths = [p for p in glob.iglob('%s%s/**/*.jpg' % 
                                   (data_stempath, which_set), recursive=True)]
    if shuffle:
        rgen = np.random.RandomState(seed)
        paths = rgen.shuffle(img_paths)
    
    idx, file_idx = 0, 1
    data = np.zeros((archive_size, height, width, channels), dtype=np.uint8)
    labels = np.zeros(archive_size, dtype=np.uint8)
    for path in img_paths:
        if idx >= archive_size - 1:
            idx = 0
            savepath = os.path.join('%snpz' % data_stempath, '%s_%d.npz' % 
                                    (which_set, file_idx))
            file_idx += 1
            np.savez(savepath, data=data, labels=labels)

        label = int(os.path.basename(os.path.dirname(path)))
        image = mpimg.imread(path)
        
        if len(image.shape) == 2:
            data[idx] = image[:, :, np.newaxis]
        labels[idx] = label
        idx += 1

images_to_pickles(which_set='train', shuffle=True, seed=1)
images_to_pickles(which_set='valid', shuffle=True, seed=1)
images_to_pickles(which_set='test', shuffle=True, seed=1)

os.listdir('mnist_npz')

data = np.load('mnist_npz/test_1.npz')
print(data['data'].shape)
print(data['labels'].shape)

plt.imshow(data['data'][0][:, :, -1], cmap='binary');
print('Class label:', data['labels'][0])

class BatchLoader():
    def __init__(self, minibatches_path, 
                 normalize=True):
        
        self.normalize = normalize

        self.train_batchpaths = [os.path.join(minibatches_path, f)
                                 for f in os.listdir(minibatches_path)
                                 if 'train' in f]
        self.valid_batchpaths = [os.path.join(minibatches_path, f)
                                 for f in os.listdir(minibatches_path)
                                 if 'valid' in f]
        self.test_batchpaths = [os.path.join(minibatches_path, f)
                                 for f in os.listdir(minibatches_path)
                                 if 'train' in f]

        self.num_train = 45000
        self.num_valid = 5000
        self.num_test = 10000
        self.n_classes = 10


    def load_train_epoch(self, batch_size=50, onehot=False,
                         shuffle_within=False, shuffle_paths=False,
                         seed=None):
        for batch_x, batch_y in self._load_epoch(which='train',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_within=shuffle_within,
                                                 shuffle_paths=shuffle_paths,
                                                 seed=seed):
            yield batch_x, batch_y

    def load_test_epoch(self, batch_size=50, onehot=False,
                        shuffle_within=False, shuffle_paths=False, 
                        seed=None):
        for batch_x, batch_y in self._load_epoch(which='test',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_within=shuffle_within,
                                                 shuffle_paths=shuffle_paths,
                                                 seed=seed):
            yield batch_x, batch_y
            
    def load_validation_epoch(self, batch_size=50, onehot=False,
                         shuffle_within=False, shuffle_paths=False, 
                         seed=None):
        for batch_x, batch_y in self._load_epoch(which='valid',
                                                 batch_size=batch_size,
                                                 onehot=onehot,
                                                 shuffle_within=shuffle_within,
                                                 shuffle_paths=shuffle_paths,
                                                 seed=seed):
            yield batch_x, batch_y

    def _load_epoch(self, which='train', batch_size=50, onehot=False,
                    shuffle_within=True, shuffle_paths=True, seed=None):

        if which == 'train':
            paths = self.train_batchpaths
        elif which == 'valid':
            paths = self.valid_batchpaths
        elif which == 'test':
            paths = self.test_batchpaths
        else:
            raise ValueError('`which` must be "train" or "test". Got %s.' %
                             which)
            
        rgen = np.random.RandomState(seed)
        if shuffle_paths:
            paths = rgen.shuffle(paths)

        for batch in paths:

            dct = np.load(batch)

            if onehot:
                labels = (np.arange(self.n_classes) == 
                          dct['labels'][:, None]).astype(np.uint8)
            else:
                labels = dct['labels']

            if self.normalize:
                # normalize to [0, 1] range
                data = dct['data'].astype(np.float32) / 255.
            else:
                data = dct['data']

            arrays = [data, labels]
            del dct
            indices = np.arange(arrays[0].shape[0])

            if shuffle_within:
                rgen.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - batch_size + 1,
                                   batch_size):
                index_slice = indices[start_idx:start_idx + batch_size]
                yield (ary[index_slice] for ary in arrays)

batch_loader = BatchLoader(minibatches_path='./mnist_npz/', 
                           normalize=True)

for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=50, onehot=True):
    print(batch_x.shape)
    print(batch_y.shape)
    break

cnt = 0
for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=50, onehot=True):
    cnt += batch_x.shape[0]
    
print('One training epoch contains %d images' % cnt)

def one_epoch():
    for batch_x, batch_y in batch_loader.load_train_epoch(batch_size=50, onehot=True):
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

batch_loader = BatchLoader(minibatches_path='./mnist_npz/', 
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

