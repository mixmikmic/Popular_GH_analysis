import os
import boto3
import sagemaker
from sagemaker.mxnet import MXNet
from mxnet import gluon
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()

gluon.data.vision.MNIST('./data/train', train=True)
gluon.data.vision.MNIST('./data/test', train=False)

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/mnist')

get_ipython().system("cat 'mnist.py'")

m = MXNet("mnist.py", 
          role=role, 
          train_instance_count=1, 
          train_instance_type="ml.c4.xlarge",
          hyperparameters={'batch_size': 100, 
                         'epochs': 20, 
                         'learning_rate': 0.1, 
                         'momentum': 0.9, 
                         'log_interval': 100})

m.fit(inputs)

predictor = m.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')

from IPython.display import HTML
HTML(open("input.html").read())

response = predictor.predict(data)
print int(response)

sagemaker.Session().delete_endpoint(predictor.endpoint)

