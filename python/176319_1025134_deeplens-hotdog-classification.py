get_ipython().run_cell_magic('bash', '', 'conda install scikit-image')

from __future__ import print_function
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from collections import OrderedDict
import skimage.io as io
import numpy as np

import mxnet as mx

from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

# Get pretrained SqueezeNet
net = models.squeezenet1_1(pretrained=True, prefix='deep_dog_')

# hot dog happens to be a class in imagenet, which this model was trained on
# we can reuse the weight for that class for better performance
# here's the index for that class
imagenet_hotdog_index = 713

# Create the model with a two class output classifier and apply the pretrained weights
deep_dog_net = models.squeezenet1_1(prefix='deep_dog_', classes=2)
deep_dog_net.collect_params().initialize()
deep_dog_net.features = net.features

# Lets take a look at what this network looks like
print(deep_dog_net)

from skimage.color import rgba2rgb

def classify_hotdog(net, url):

    # Pull in image and ensure there are only 3 color channels (RGB)
    I = io.imread(url)
    if I.shape[2] == 4:
        I = rgba2rgb(I)
        
    # Normalize the color channels and crop the image to the expected input size (224,224)
    image = mx.nd.array(I).astype(np.uint8)
    image = mx.image.resize_short(image, 256)
    image, _ = mx.image.center_crop(image, (224, 224))
    image = mx.image.color_normalize(image.astype(np.float32)/255,
                                     mean=mx.nd.array([0.485, 0.456, 0.406]),
                                     std=mx.nd.array([0.229, 0.224, 0.225]))

    # Flip the color channels from RGB to the expected BGR input
    image = mx.nd.transpose(image.astype('float32'), (2,1,0))
    image = mx.nd.expand_dims(image, axis=0)
    
    # Feed the pre-processed image into the net and get the predicted result
    inference_result = net(image)
    print('Raw inference output is:'+str(inference_result))
    
    # Squeeze the inference result into a softmax function to turn it into a probability
    out = mx.nd.SoftmaxActivation(inference_result)
    print('Probabilities are: '+str(out[0].asnumpy()))
    
    # Take max probability to predict if the image has a hotdog or not
    result = np.argmax(out.asnumpy())
    outstring = ['Not hotdog!', 'Hotdog!']
    print(outstring[result])

get_ipython().run_cell_magic('bash', '', 'wget http://www.wienerschnitzel.com/wp-content/uploads/2014/10/hotdog_mustard-main.jpg\nwget https://www.what-dog.net/Images/faces2/scroll001.jpg')

deep_dog_net.hybridize()

# Let's run the classification on our tow downloaded images to see what our model comes up with
classify_hotdog(deep_dog_net, './hotdog_mustard-main.jpg') # check for hotdog
classify_hotdog(deep_dog_net, './scroll001.jpg') # check for not hotdog

from mxnet.test_utils import download

# Pull the new parameters using the download utility provided by MXNet
download('https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/models/deep-dog-5a342a6f.params',
         overwrite=True)

# This simply applies the new parameters onto the model we already have
deep_dog_net.load_params('deep-dog-5a342a6f.params', mx.cpu())

deep_dog_net.hybridize()
classify_hotdog(deep_dog_net, './hotdog_mustard-main.jpg')
classify_hotdog(deep_dog_net, './scroll001.jpg')

deep_dog_net.export('hotdog_or_not_model')

import boto3
import re

assumed_role = boto3.client('sts').get_caller_identity()['Arn']
s3_access_role = re.sub(r'^(.+)sts::(\d+):assumed-role/(.+?)/.*$', r'\1iam::\2:role/\3', assumed_role)
print(s3_access_role)
s3 = boto3.resource('s3')

json = open('hotdog_or_not_model-symbol.json', 'rb')
params = open('hotdog_or_not_model-0000.params', 'rb')
s3.Bucket('test-bucket').put_object(Key='hotdog_or_not_model-symbol.json', Body=json)
s3.Bucket('test-bucket').put_object(Key='hotdog_or_not_model-0000.params', Body=params)

