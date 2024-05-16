import collections
import glob
import io
import sys
import six
from six.moves import cPickle, xrange

from lasagne.utils import floatX
import numpy as np
import lasagne, theano

#To enable the GPU, run the following code
import theano.gpuarray
theano.config.floatX='float32'
theano.gpuarray.use('cuda')

# vgg16 includes the model definition and function to read and preprocess images from VGG16
from vgg16 import build_model, prep_image

# Populating the interactive namespace from numpy and matplotlib
get_ipython().magic('pylab inline')

def distance_mat(x, m, p=2.0):
    """ Compute the L-p distance between a feature vector `x`
    and a matrix of feature vectors `x`.
    """
    diff = (np.abs(x - m)**p).sum(axis=1)**(1.0/p)
    return diff

def knn_idx(x, features, p=2):
    """Return the row index of the most similar features compared to `x`."""
    dist = distance_mat(x, features, p=p)
    return np.argmin(dist)

class1_dir = './dog/'
class1_name = 'dog'
class2_dir = './donut/'
class2_name = 'donut'
test_dir = './test/'

# List files under the "dog/" directory
class1_files = glob.glob(class1_dir + '*')
# Load the images
class1_images = [plt.imread(io.BytesIO(open(f, 'rb').read()), f.split('.')[-1]) for f in class1_files]
# Build the target classes
class1_targets = [class1_name] * len(class1_files)

# Do the same for the second class
class2_files = glob.glob(class2_dir + '*')
class2_images = [plt.imread(io.BytesIO(open(f, 'rb').read()), f.split('.')[-1]) for f in class2_files]
class2_targets = [class2_name] * len(class2_files)

# Create the dataset by combining both classes
train_files = class1_files + class2_files
train_images = class1_images + class2_images
train_targets = class1_targets + class2_targets

# Read the test files
test_files = glob.glob(test_dir + '*')
test_images = [plt.imread(io.BytesIO(open(f, 'rb').read()), f.split('.')[-1]) for f in test_files]

# vgg16.pkl contains the trained weights and the mean values needed for the preprocessing.
with open('vgg16.pkl', 'rb') as f:
    if six.PY3:
        d = cPickle.load(f, encoding='latin1')
    else:
        d = cPickle.load(f)

MEAN_IMAGE = d['mean value']
# Get the Lasagne model
net = build_model()
# Set the pre-trained weights
lasagne.layers.set_all_param_values(net['prob'], d['param values'])

# The different layer outputs you can reuse for the prediction
print(net.keys())

# Get the graph that computes the last feature layers (fc8) of the model
# deterministic=True makes the Dropout layers do nothing as we don't train it
output = lasagne.layers.get_output(net['fc8'], deterministic=True)
# Compile the Theano function to be able to execute it.
compute_last = theano.function([net['input'].input_var], output)

def compute_feats(images):
    """Compute the features of many images."""
    preps = []
    for img in images:
        # prep_image returns a 4d tensor with only 1 image
        # remove the first dimensions to batch them ourself
        preps.append(prep_image(img, MEAN_IMAGE)[1][0])
    # Batch compute the features.
    return compute_last(preps)


# Compute the features of the train and test datasets
train_feats = compute_feats(train_images)
test_feats = compute_feats(test_images)

# Show the name of the file corresponding to example 0
print(test_files[0])

# Call knn_idx to get the nearest neighbor of this example
idx0 = knn_idx(test_feats[0], train_feats)

# Show the name of this training file
print(train_files[idx0])

# Show the predicted class
print(train_targets[idx0])

def most_frequent(label_list):
    return collections.Counter(label_list).most_common()[0][0]

def knn_idx(x, features, p=2, k=1):
    dist = distance_mat(x, features, p=p)
    return np.argsort(dist)[:k]


def plot_knn(test_image, test_feat, train_images, train_feats, train_classes, k=1):
    knn_i = knn_idx(test_feat, train_feats, k=k)
    knn_images = [train_images[i] for i in knn_i]
    knn_classes = [train_classes[i] for i in knn_i]
    pred_class = most_frequent(knn_classes)
    figure(figsize=(12, 4))
    subplot(1, k+2, 1)
    imshow(prep_image(test_image, MEAN_IMAGE)[0])
    axis('off')
    title('prediction : ' + pred_class)
    for i in xrange(k):
        knn_preproc = prep_image(knn_images[i], MEAN_IMAGE)[0]
        subplot(1, k+2, i+3)
        imshow(knn_preproc)
        axis('off')
        title(knn_classes[i])

for i in range(len(test_images)):
    plot_knn(test_images[i], test_feats[i], train_images, train_feats, train_targets, k=7)



