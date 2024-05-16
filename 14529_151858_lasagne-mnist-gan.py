import theano
import numpy as np
import matplotlib.pylab as plt
import csv, os, random, sys

import lasagne
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import Conv2DLayer, InputLayer, ConcatLayer
from lasagne.layers import DenseLayer, Pool2DLayer, FlattenLayer

print "theano",theano.version.full_version
print "lasagne",lasagne.__version__



#Set seed for random numbers:
np.random.seed(1234)
lasagne.random.set_rng(np.random.RandomState(1234))



if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

import gzip
def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)
    data = np.asarray([np.rot90(np.fliplr(x[0])) for x in data])
    data = data.reshape(-1, 1, 28, 28)
    return data / np.float32(255)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

x_train = load_mnist_images('train-images-idx3-ubyte.gz')
t_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
t_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
x_train, x_val = x_train[:-10000], x_train[-10000:]
t_train, t_val = t_train[:-10000], t_train[-10000:]



get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 5)
plt.imshow(x_train[0][0],interpolation='none', cmap='gray');



num_units = 1024
encoder_size = 100
noise_size = 100



import theano
import theano.tensor as T
import lasagne
import lasagne.layers
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import InputLayer, Conv2DLayer, Pool2DLayer, DenseLayer, MaxPool2DLayer, Upscale2DLayer
from lasagne.layers import ConcatLayer, DropoutLayer, ReshapeLayer, TransposedConv2DLayer

gen_input_var = T.matrix("gen_input_var")

gen = lasagne.layers.InputLayer(shape=(None, noise_size),input_var=gen_input_var)
print gen.output_shape

gen = lasagne.layers.ReshapeLayer(gen, (-1, noise_size, 1, 1))
print gen.output_shape
gen = BatchNormLayer(TransposedConv2DLayer(gen, num_filters=num_units, filter_size=4, stride=1))
print gen.output_shape

gen = BatchNormLayer(TransposedConv2DLayer(gen, num_filters=num_units/2, filter_size=8, stride=1))
print gen.output_shape

gen = BatchNormLayer(TransposedConv2DLayer(gen, num_filters=num_units/4, filter_size=16, stride=1))
print gen.output_shape

gen = TransposedConv2DLayer(gen, num_filters=num_units/8, filter_size=6)
print gen.output_shape

gen = Conv2DLayer(gen, num_filters=1, filter_size=4, nonlinearity=lasagne.nonlinearities.sigmoid)
print gen.output_shape

lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

disc = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=None)
print disc.output_shape

disc = BatchNormLayer(Conv2DLayer(disc, num_filters=num_units/4, filter_size=5, stride=2, pad=2, nonlinearity=lrelu))
print disc.output_shape

disc = BatchNormLayer(Conv2DLayer(disc, num_filters=num_units/2, filter_size=5, stride=2, pad=2, nonlinearity=lrelu))
print disc.output_shape

disc = (Conv2DLayer(disc, num_filters=num_units, filter_size=5, stride=2, pad=2, nonlinearity=lrelu))
print disc.output_shape

disc = FlattenLayer(disc)
disc = DenseLayer(disc, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
print disc.output_shape





# create train functions 
lr = theano.shared(np.array(0., dtype=theano.config.floatX))

gen_output = lasagne.layers.get_output(gen)

one = T.constant(1., dtype=theano.config.floatX)
input_real = T.tensor4('target')


disc_output_fake = lasagne.layers.get_output(disc, inputs=gen_output)
disc_output_real = lasagne.layers.get_output(disc, inputs=input_real)
disc_loss = -(T.log(disc_output_real) + T.log(one-disc_output_fake)).mean()
disc_params = lasagne.layers.get_all_params(disc, trainable=True)
disc_updates = lasagne.updates.adam(disc_loss, disc_params, learning_rate=lr, beta1=0.5)


gen_loss = -T.log(disc_output_fake).mean()
gen_params = lasagne.layers.get_all_params(gen, trainable=True)
gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate=lr, beta1=0.5)


print "Computing functions"

gen_fn = theano.function([gen_input_var], gen_output, allow_input_downcast=True)

train_gen_fn = theano.function([gen_input_var], 
                               [gen_loss],
                               updates=gen_updates, 
                               allow_input_downcast=True)

disc_fn = theano.function([input_real], disc_output_real, allow_input_downcast=True)

train_disc_fn = theano.function([gen_input_var, input_real], 
                                [disc_loss],
                                updates=disc_updates,
                                allow_input_downcast=True)

print "Done"



noise = np.random.uniform(size=(10, noise_size))
img = gen_fn(noise)
print img.shape

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (20, 5)
plt.imshow(np.concatenate(img[:,0], axis=1),interpolation='none', cmap='gray');



disc_fn(img)

disc_fn(x_train[:10])





noise = np.random.normal(size=(1, noise_size))
train_gen_fn(noise)



train_disc_fn(noise, x_train[:1])





from scipy.ndimage.filters import gaussian_filter
def permute(img):
        
    rotation = random.choice([True, False])
    flip = random.choice([True, False])
    blur = random.randint(0,1)
    pad = random.randint(0,5)
    
    if (pad != 0):
        img = img[:,pad:-pad,pad:-pad]
        img = [scipy.misc.imresize(img[0], image_size), 
               scipy.misc.imresize(img[1], image_size), 
               scipy.misc.imresize(img[2], image_size)]
    
    if (blur != 0):
        img = gaussian_filter(img, sigma=blur)
    
    #rotate 
    if rotation:
        img = [np.rot90(img[0], 2),np.rot90(img[1], 2) ,np.rot90(img[2], 2)]
    
    #flip 
    if (flip):
        img = np.fliplr(img)  
    
    return img



num_samples = 1000



# ### D
# lr.set_value(0.0001)
# for i in range(10):
#     errs = []
#     for i in range(10):
#         noise = np.random.normal(size=(num_samples, noise_size))
#         samples = np.random.randint(0,len(x_train),num_samples)
#         err = train_disc_fn(noise, x_train[samples])
#         errs.append(err)
#     print "d",np.mean(errs)



# ### G
# lr.set_value(0.001)
# for i in range(1000):
#     errs = []
#     for i in range(5):
#         noise = np.random.normal(size=(num_samples, noise_size))
#         err = train_gen_fn(noise)
#         errs.append(err)
#     print "g",np.mean(errs)



lr.set_value(0.0001)
for j in range(100):
    err = 1
    while err > 0.5:
        noise = np.random.normal(size=(num_samples, noise_size))
        samples = np.random.randint(0,len(x_train),num_samples)
        err = train_disc_fn(noise, x_train[samples])[0]
    print "d",err

    err = 1
    while err > 0.5:
        noise = np.random.normal(size=(num_samples, noise_size))
        err = train_gen_fn(noise)[0]
    print "g",err



noise = np.random.uniform(size=(10, noise_size))
img = gen_fn(noise)
print img.shape

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (20, 5)
plt.imshow(np.concatenate(img[:,0], axis=1),interpolation='none', cmap='gray');











