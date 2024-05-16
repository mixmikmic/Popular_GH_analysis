# Rather than importing everything manually, we'll make things easy
#   and load them all in utils.py, and just import them from there.
get_ipython().magic('matplotlib inline')
import utils; reload(utils)
from utils import *

get_ipython().magic('matplotlib inline')
from __future__ import division,print_function
import os, json
from glob import glob
import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots, get_batches, plot_confusion_matrix, get_data

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

#path = "../data/dogsandcats_small/" # we copied a fraction of the full set for tests
path = "../data/dogsandcats/"
model_path = path + "models/"
if not os.path.exists(model_path):
    os.mkdir(model_path)
    print('Done')

from vgg16 import Vgg16

batch_size = 100

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, 
                batch_size=batch_size, class_mode='categorical'):
    return gen.flow_from_directory(path+dirname, target_size=(224,224), 
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

# Use batch size of 1 since we're just doing preprocessing on the CPU
val_batches = get_batches('valid', shuffle=False, batch_size=batch_size) # no shuffle as we store conv output
trn_batches = get_batches('train', shuffle=False, batch_size=batch_size) # no shuffle as we store conv output

val_batches.filenames[0:10]

val_labels = onehot(val_batches.classes)
trn_labels = onehot(trn_batches.classes)

# DONT USE IT FOR NOW
if False:
    realvgg = Vgg16()
    conv_layers, fc_layers = split_at(realvgg.model, Convolution2D)
    conv_model = Sequential(conv_layers)

vggbase = Vgg16()
vggbase.model.pop()
vggbase.model.pop()

# DONT USE IT FOR NOW
if False:
    try:
        val_features = load_array(model_path+'valid_convlayer_features.bc')
        if False: # force update
            raise
    except:
        print('Missing file')
        val_features = conv_model.predict_generator(val_batches, val_batches.nb_sample)
        save_array(model_path + 'valid_convlayer_features.bc', val_features)

try:
    val_vggfeatures = load_array(model_path+'valid_vggbase_features.bc')
    if False: # force update
        raise
except:
    print('Missing file')
    val_vggfeatures = vggbase.model.predict_generator(val_batches, val_batches.nb_sample)
    save_array(model_path + 'valid_vggbase_features.bc', val_vggfeatures)

# DONT USE IT FOR NOW
if False:
    try:
        trn_features = load_array(model_path+'train_convlayer_features.bc')
        if False: # force update
            raise
    except:
        print('Missing file')
        trn_features = conv_model.predict_generator(trn_batches, trn_batches.nb_sample)
        save_array(model_path + 'train_convlayer_features.bc', trn_features)

try:
    trn_vggfeatures = load_array(model_path+'train_vggbase_features.bc')
    if False: # force update
        raise
except:
    print('Missing file')
    trn_vggfeatures = vggbase.model.predict_generator(trn_batches, trn_batches.nb_sample)
    save_array(model_path + 'train_vggbase_features.bc', trn_vggfeatures)

ll_layers = [BatchNormalization(input_shape=(4096,)),
             Dropout(0.25),
             Dense(2, activation='softmax')]
ll_model = Sequential(ll_layers)
ll_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

ll_model.optimizer.lr = 10*1e-5
ll_model.fit(trn_vggfeatures, trn_labels, validation_data=(val_vggfeatures, val_labels), nb_epoch=5)

#ll_model.save_weights(model_path+'llmodel_finetune1.h5')
#ll_model.load_weights(model_path+'llmodel_finetune1.h5')

test_batches = get_batches('test', shuffle=False, batch_size=batch_size, class_mode=None)
testfiles = test_batches.filenames
testfiles[0:10]

try:
    test_vggfeatures = load_array(model_path+'test_vggbase_features.bc')
    if False: # force update
        raise
except:
    print('Missing file')
    test_vggfeatures = vggbase.model.predict_generator(test_batches, test_batches.nb_sample)
    save_array(model_path + 'test_vggbase_features.bc', test_vggfeatures)

test_preds = ll_model.predict_on_batch(test_vggfeatures)

assert(len(test_preds) == 12500)

test_preds[0:10]

dog_idx = 1
Z1 = [{'id':int(f.split('/')[-1].split('.')[0]), 'label':min(max(round(p[dog_idx],5),0.0001),0.9999)} 
      for f, p in zip(testfiles, test_preds)]
def comp(x,y):
    return int(x['id']) - int(y['id'])
Z1 = sorted(Z1, comp)
Z1[0:18]

import csv
        
with open('predictions.csv', 'w') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for z in Z1:
        writer.writerow(z)





