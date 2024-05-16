import theano
import theano.tensor as T

import lasagne
from lasagne.utils import floatX

import numpy as np
import scipy

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os # for directory listings
import pickle
import time

AS_PATH='./images/art-style'

from model import googlenet

net = googlenet.build_model()
net_input_var = net['input'].input_var
net_output_layer = net['prob']

params = pickle.load(open('./data/googlenet/blvc_googlenet.pkl', 'rb'), encoding='iso-8859-1')
model_param_values = params['param values']
#classes = params['synset words']
lasagne.layers.set_all_param_values(net_output_layer, model_param_values)

IMAGE_W=224
print("Loaded Model parameters")

photos = [ '%s/photos/%s' % (AS_PATH, f) for f in os.listdir('%s/photos/' % AS_PATH) if not f.startswith('.')]
photo_i=-1 # will be incremented in next cell (i.e. to start at [0])

photo_i += 1
photo = plt.imread(photos[photo_i % len(photos)])
photo_rawim, photo = googlenet.prep_image(photo)
plt.imshow(photo_rawim)

styles = [ '%s/styles/%s' % (AS_PATH, f) for f in os.listdir('%s/styles/' % AS_PATH) if not f.startswith('.')]
style_i=-1 # will be incremented in next cell (i.e. to start at [0])

style_i += 1
art = plt.imread(styles[style_i % len(styles)])
art_rawim, art = googlenet.prep_image(art)
plt.imshow(art_rawim)

def plot_layout(combined):
    def no_axes():
        plt.gca().xaxis.set_visible(False)    
        plt.gca().yaxis.set_visible(False)    
        
    plt.figure(figsize=(9,6))

    plt.subplot2grid( (2,3), (0,0) )
    no_axes()
    plt.imshow(photo_rawim)

    plt.subplot2grid( (2,3), (1,0) )
    no_axes()
    plt.imshow(art_rawim)

    plt.subplot2grid( (2,3), (0,1), colspan=2, rowspan=2 )
    no_axes()
    plt.imshow(combined, interpolation='nearest')

    plt.tight_layout()

def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g

def content_loss(P, X, layer):
    p = P[layer]
    x = X[layer]
    
    loss = 1./2 * ((x - p)**2).sum()
    return loss

def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]
    
    A = gram_matrix(a)
    G = gram_matrix(x)
    
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss

def total_variation_loss(x):
    return (((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25).sum()

layers = [
    # used for 'content' in photo - a mid-tier convolutional layer 
    'inception_4b/output', 
    
    # used for 'style' - conv layers throughout model (not same as content one)
    'conv1/7x7_s2', 'conv2/3x3', 'inception_3b/output', 'inception_4d/output',
]
#layers = [
#    # used for 'content' in photo - a mid-tier convolutional layer 
#    'pool4/3x3_s2', 
#    
#    # used for 'style' - conv layers throughout model (not same as content one)
#    'conv1/7x7_s2', 'conv2/3x3', 'pool3/3x3_s2', 'inception_5b/output',
#]
layers = {k: net[k] for k in layers}

input_im_theano = T.tensor4()
outputs = lasagne.layers.get_output(layers.values(), input_im_theano)

photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                  for k, output in zip(layers.keys(), outputs)}
art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                for k, output in zip(layers.keys(), outputs)}

# Get expressions for layer activations for generated image
generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

gen_features = lasagne.layers.get_output(layers.values(), generated_image)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}

losses = []

# content loss
cl = 10 /1000.
losses.append(cl * content_loss(photo_features, gen_features, 'inception_4b/output'))

# style loss
sl = 20 *1000.
losses.append(sl * style_loss(art_features, gen_features, 'conv1/7x7_s2'))
losses.append(sl * style_loss(art_features, gen_features, 'conv2/3x3'))
losses.append(sl * style_loss(art_features, gen_features, 'inception_3b/output'))
losses.append(sl * style_loss(art_features, gen_features, 'inception_4d/output'))
#losses.append(sl * style_loss(art_features, gen_features, 'inception_5b/output'))

# total variation penalty
vp = 0.01 /1000. /1000.
losses.append(vp * total_variation_loss(generated_image))

total_loss = sum(losses)

grad = T.grad(total_loss, generated_image)

# Theano functions to evaluate loss and gradient - takes around 1 minute (!)
f_loss = theano.function([], total_loss)
f_grad = theano.function([], grad)

# Helper functions to interface with scipy.optimize
def eval_loss(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return f_loss().astype('float64')

def eval_grad(x0):
    x0 = floatX(x0.reshape((1, 3, IMAGE_W, IMAGE_W)))
    generated_image.set_value(x0)
    return np.array(f_grad()).flatten().astype('float64')

generated_image.set_value(photo)
#generated_image.set_value(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))

x0 = generated_image.get_value().astype('float64')
iteration=0

t0 = time.time()

scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=40) 

x0 = generated_image.get_value().astype('float64')
iteration += 1

if False:
    plt.figure(figsize=(8,8))
    plt.imshow(googlenet.deprocess(x0), interpolation='nearest')
    plt.axis('off')
    plt.text(270, 25, '# {} in {:.1f}sec'.format(iteration, (float(time.time() - t0))), fontsize=14)
else:
    plot_layout(googlenet.deprocess(x0))
    print('Iteration {}, ran in {:.1f}sec'.format(iteration, float(time.time() - t0)))



