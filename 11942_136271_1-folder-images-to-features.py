import os

import tensorflow.contrib.keras as keras
import numpy as np

import datetime
t_start=datetime.datetime.now()

import pickle

image_folder_path = './data/Flickr30k/flickr30k-images'

output_dir = './data/cache'

output_filepath = os.path.join(output_dir, 
                                'FEATURES_%s_%s.pkl' % ( 
                                 image_folder_path.replace('./', '').replace('/', '_'),
                                 t_start.strftime("%Y-%m-%d_%H-%M"),
                                ), )
output_filepath

from tensorflow.contrib.keras.api.keras.applications.inception_v3 import decode_predictions
from tensorflow.contrib.keras.api.keras.preprocessing import image as keras_preprocessing_image

from tensorflow.contrib.keras.api.keras.applications.inception_v3 import InceptionV3, preprocess_input

BATCHSIZE=16

model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
print("InceptionV3 loaded")

import re
good_image = re.compile( r'\.(jpg|png|gif)$', flags=re.IGNORECASE )

img_arr = [ f for f in os.listdir(image_folder_path) if good_image.search(f) ]
', '.join( img_arr[:3] ), ', '.join( img_arr[-3:] )

# Create a generator for preprocessed images
def preprocessed_image_gen():
    #target_size=model.input_shape[1:]
    target_size=(299, 299, 3)
    print("target_size", target_size)
    for img_name in img_arr:
        #print("img_name", img_name)
        img_path = os.path.join(image_folder_path, img_name)
        img = keras_preprocessing_image.load_img(img_path, target_size=target_size)
        yield keras.preprocessing.image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)  # This is to make a single image into a suitable array

def image_batch(batchsize=BATCHSIZE):
    while True:  # This needs to run 'for ever' for Keras input, even if only a fixed number are required
        preprocessed_image_generator = preprocessed_image_gen()
        start = True
        for img in preprocessed_image_generator:
            if start:
                arr, n, start = [], 0, False
            arr.append(img)
            n += 1
            if n>=batchsize: 
                stack = np.stack( arr, axis=0 )
                #print("stack.shape", stack.shape)
                preprocessed = preprocess_input( stack )
                #print("preprocessed.shape", preprocessed.shape)
                yield preprocessed
                start=True
        if len(arr)>0:
            stack = np.stack( arr, axis=0 )
            print("Final stack.shape", stack.shape)
            preprocessed = preprocess_input( stack )
            print("Final preprocessed.shape", preprocessed.shape)
            yield preprocessed

if False:
    image_batcher = image_batch()
    batch = next(image_batcher)
    features = model.predict_on_batch(batch)
    features.shape

# This should do the batch creation on the CPU and the analysis on the GPU asynchronously.
import math  # for ceil

t0=datetime.datetime.now()

features = model.predict_generator(image_batch(), steps = math.ceil( len(img_arr)/BATCHSIZE) )  #, verbose=1

features.shape, (datetime.datetime.now()-t0)/len(img_arr)*1000.

# Save the data into a useful structure

save_me = dict(
    features = features,
    img_arr = img_arr,
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
with open( output_filepath, 'wb') as f:
    pickle.dump(save_me, f)
    
print("Features saved to '%s'" %(output_filepath,))



