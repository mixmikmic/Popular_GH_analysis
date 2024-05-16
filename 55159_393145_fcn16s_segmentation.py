get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import math
import copy

import skimage.io as io
from scipy.misc import bytescale

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import merge

from utils import fcn32_blank, fcn_32s_to_16s, prediction

image_size = 64*8 # INFO: initially tested with 256, 448, 512

fcn32model = fcn32_blank(image_size)

#fcn32model.summary() # visual inspection of model architecture

fcn16model = fcn_32s_to_16s(fcn32model)

# INFO : dummy image array to test the model passes
imarr = np.ones((3,image_size,image_size))
imarr = np.expand_dims(imarr, axis=0)

#testmdl = Model(fcn32model.input, fcn32model.layers[10].output) # works fine
testmdl = fcn16model # works fine
testmdl.predict(imarr).shape

if (testmdl.predict(imarr).shape != (1,21,image_size,image_size)):
    print('WARNING: size mismatch will impact some test cases')

fcn16model.summary() # visual inspection of model architecture

from scipy.io import loadmat

data = loadmat('pascal-fcn16s-dag.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
p = data['params']
description = data['meta'][0,0].classes[0,0].description

l.shape, p.shape, description.shape

class2index = {}
for i, clname in enumerate(description[0,:]):
    class2index[str(clname[0])] = i
    
print(sorted(class2index.keys()))

if False: # inspection of data structure
    print(dir(l[0,31].block[0,0]))
    print(dir(l[0,36].block[0,0]))

for i in range(0, p.shape[1]-1, 2):
    print(i,
          str(p[0,i].name[0]), p[0,i].value.shape,
          str(p[0,i+1].name[0]), p[0,i+1].value.shape)

for i in range(l.shape[1]):
    print(i,
          str(l[0,i].name[0]), str(l[0,i].type[0]),
          [str(n[0]) for n in l[0,i].inputs[0,:]],
          [str(n[0]) for n in l[0,i].outputs[0,:]])

# documentation for the dagnn.Crop layer :
# https://github.com/vlfeat/matconvnet/blob/master/matlab/%2Bdagnn/Crop.m

def copy_mat_to_keras(kmodel):
    
    kerasnames = [lr.name for lr in kmodel.layers]

    prmt = (3,2,0,1) # WARNING : important setting as 2 of the 4 axis have same size dimension
    
    for i in range(0, p.shape[1]-1, 2):
        matname = '_'.join(p[0,i].name[0].split('_')[0:-1])
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            print 'found : ', (str(matname), kindex)
            l_weights = p[0,i].value
            l_bias = p[0,i+1].value
            f_l_weights = l_weights.transpose(prmt)
            f_l_weights = np.flip(f_l_weights, 2)
            f_l_weights = np.flip(f_l_weights, 3)
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
        else:
            print 'not found : ', str(matname)

#copy_mat_to_keras(fcn32model)
copy_mat_to_keras(fcn16model)

im = Image.open('rgb.jpg') # http://www.robots.ox.ac.uk/~szheng/crfasrnndemo/static/rgb.jpg
im = im.crop((0,0,319,319)) # WARNING : manual square cropping
im = im.resize((image_size,image_size))

plt.imshow(np.asarray(im))

crpim = im # WARNING : we deal with cropping in a latter section, this image is already fit
preds = prediction(fcn16model, crpim, transform=True)

#imperson = preds[0,class2index['person'],:,:]
imclass = np.argmax(preds, axis=1)[0,:,:]

plt.figure(figsize = (15, 7))
plt.subplot(1,3,1)
plt.imshow( np.asarray(crpim) )
plt.subplot(1,3,2)
plt.imshow( imclass )
plt.subplot(1,3,3)
plt.imshow( np.asarray(crpim) )
masked_imclass = np.ma.masked_where(imclass == 0, imclass)
#plt.imshow( imclass, alpha=0.5 )
plt.imshow( masked_imclass, alpha=0.5 )

# List of dominant classes found in the image
for c in np.unique(imclass):
    print c, str(description[0,c][0])

bspreds = bytescale(preds, low=0, high=255)

plt.figure(figsize = (15, 7))
plt.subplot(2,3,1)
plt.imshow(np.asarray(crpim))
plt.subplot(2,3,3+1)
plt.imshow(bspreds[0,class2index['background'],:,:], cmap='seismic')
plt.subplot(2,3,3+2)
plt.imshow(bspreds[0,class2index['person'],:,:], cmap='seismic')
plt.subplot(2,3,3+3)
plt.imshow(bspreds[0,class2index['bicycle'],:,:], cmap='seismic')



