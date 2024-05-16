from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Some of these are hard to distinguish.
# Check https://quickdraw.withgoogle.com/data for examples
zoo = ['frog', 'horse', 'lion', 'monkey', 'octopus', 'owl', 'rhinoceros', 
       'snail', 'tiger', 'zebra']

# Mapping between category names and ids
animal2id = dict((c,i) for i,c in enumerate(zoo))
id2animal = dict((i,c) for i,c in enumerate(zoo))
for i, animal in id2animal.items():
    print("Class {}: {}".format(i, animal))

from six.moves.urllib.request import urlretrieve
import os

DATA_DIR = 'data/'

def maybe_download(url, data_dir):
    filename = url.split('/')[-1]
    file_path = os.path.join(data_dir, filename)

    # Check if the file already exists.
    if not os.path.exists(file_path):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        print("Downloading {} to {}".format(url, file_path))
        file_path, _ = urlretrieve(url=url, filename=file_path)
    else:
        print("Using previously downloaded file: {}".format(file_path))
    return file_path

def load_data(file_path, max_examples=2000, example_name=''):
    d = np.load(open(file_path, 'r'))
    d = d[:max_examples,:] # limit number of instances to save memory
    print("Loaded {} {} examples of dimension {} from {}".format(
            d.shape[0], example_name, d.shape[1], file_path))
    return d

data= []
labels =[]

for animal in zoo:
    url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy".format(animal)
    file_path = maybe_download(url, DATA_DIR)
    data.append(load_data(file_path, max_examples = 1000, example_name = animal))
    labels.extend([animal2id[animal]]*data[-1].shape[0])

data = np.concatenate(data, axis=0)
labels = np.array(labels)
print("Final shape of data: {}".format(data.shape))

import matplotlib.pyplot as plt

n_samples = 10
random_indices = np.random.permutation(data.shape[0])

for i in random_indices[:n_samples]:
    print(i, labels[i])
    print("Category {}: {}".format(labels[i], id2animal[labels[i]]))

    # We'll show the image and its pixel value histogram side-by-side.

    # To interpret the values as a 28x28 image, we need to reshape
    # the numpy array, which is one dimensional.
    image = data[i, :]

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image.reshape(28, 28), cmap=plt.cm.Greys, interpolation='nearest')
    ax2.hist(image, bins=20)
    ax1.grid(False)
    plt.show()

if data.dtype == 'uint8':  # avoid doing this twice
    data = data.astype(np.float32)
    data = (data - (255 / 2.0)) / 255

random_indices = np.random.permutation(labels.shape[0])

print("Labels before:")
print(labels[random_indices[:5]])

def one_hot(labels, n_classes):
    n_labels = len(labels)
    one_hot_labels = np.zeros((n_labels, n_classes))
    one_hot_labels[np.arange(n_labels), labels] = 1
    return one_hot_labels

labels_one_hot = one_hot(labels, len(zoo))

print("Labels after:")
print(labels_one_hot[random_indices[:5]])

n_test_examples = 1000

random_indices = np.random.permutation(data.shape[0])
test_data = data[random_indices[:n_test_examples],:]
test_labels = labels_one_hot[random_indices[:n_test_examples],:]
train_data = data[random_indices[n_test_examples:],:]
train_labels = labels_one_hot[random_indices[n_test_examples:],:]
print("Data shapes: ", test_data.shape, test_labels.shape, train_data.shape, train_labels.shape)

outfile_name = os.path.join(DATA_DIR, "zoo.npz")
with open(outfile_name, 'w') as outfile:
    np.savez(outfile, train_data, train_labels, test_data, test_labels)
print ("Saved train/test data to {}".format(outfile_name))

