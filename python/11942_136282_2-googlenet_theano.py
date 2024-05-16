import theano
import theano.tensor as T

import lasagne
from lasagne.utils import floatX

import numpy as np
import scipy

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os
import json
import pickle

from model import googlenet

# Uncomment and execute this cell to see the GoogLeNet source
# %load models/imagenet_theano/googlenet.py

# !wget -N --directory-prefix=./data/googlenet https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl

cnn_layers = googlenet.build_model()
cnn_input_var = cnn_layers['input'].input_var
cnn_feature_layer = cnn_layers['loss3/classifier']
cnn_output_layer = cnn_layers['prob']

get_cnn_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_feature_layer))

print("Defined GoogLeNet model")

params = pickle.load(open('./data/googlenet/blvc_googlenet.pkl', 'rb'), encoding='iso-8859-1')
model_param_values = params['param values']
classes = params['synset words']
lasagne.layers.set_all_param_values(cnn_output_layer, model_param_values)

MEAN_VALUES = np.array([104, 117, 123]).reshape((3,1,1))

def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        #im = skimage.transform.resize(im, (224, w*224/h), preserve_range=True)
        im = scipy.misc.imresize(im, (224, w*224/h))
        
    else:
        #im = skimage.transform.resize(im, (h*224/w, 224), preserve_range=True)
        im = scipy.misc.imresize(im, (h*224/w, 224))

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES
    return rawim, floatX(im[np.newaxis])

im = plt.imread('./images/cat-with-tongue_224x224.jpg')
plt.imshow(im)

rawim, cnn_im = prep_image(im)

plt.imshow(rawim)

p = get_cnn_features(cnn_im)
print(classes[p.argmax()])

image_dir = './images/'

image_files = [ '%s/%s' % (image_dir, f) for f in os.listdir(image_dir) 
                 if (f.lower().endswith('png') or f.lower().endswith('jpg')) and f!='logo.png' ]

import time
t0 = time.time()
for i, f in enumerate(image_files):
    im = plt.imread(f)
    #print("Image File:%s" % (f,))
    rawim, cnn_im = prep_image(im)
        
    prob = get_cnn_features(cnn_im)
    top5 = np.argsort(prob[0])[-1:-6:-1]    

    plt.figure()
    plt.imshow(im.astype('uint8'))
    plt.axis('off')
    for n, label in enumerate(top5):
        plt.text(350, 50 + n * 25, '{}. {}'.format(n+1, classes[label]), fontsize=14)
        
print("DONE : %6.2f seconds each" %(float(time.time() - t0)/len(image_files),))



