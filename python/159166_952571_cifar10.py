import os
import boto3
import sagemaker
from sagemaker.mxnet import MXNet
from mxnet import gluon
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()

from cifar10_utils import download_training_data
download_training_data()

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/gluon-cifar10')
print('input spec (in this case, just an S3 path): {}'.format(inputs))

get_ipython().system("cat 'cifar10.py'")

m = MXNet("cifar10.py", 
          role=role, 
          train_instance_count=2, 
          train_instance_type="ml.p2.xlarge",
          hyperparameters={'batch_size': 128, 
                           'epochs': 50, 
                           'learning_rate': 0.1, 
                           'momentum': 0.9})

m.fit(inputs)

predictor = m.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')

# load the CIFAR10 samples, and convert them into format we can use with the prediction endpoint
from cifar10_utils import read_images

filenames = ['images/airplane1.png',
             'images/automobile1.png',
             'images/bird1.png',
             'images/cat1.png',
             'images/deer1.png',
             'images/dog1.png',
             'images/frog1.png',
             'images/horse1.png',
             'images/ship1.png',
             'images/truck1.png']

image_data = read_images(filenames)

for i, img in enumerate(image_data):
    response = predictor.predict(img)
    print('image {}: class: {}'.format(i, int(response)))

sagemaker.Session().delete_endpoint(predictor.endpoint)

