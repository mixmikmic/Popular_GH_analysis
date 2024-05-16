import boto3, re
from sagemaker import get_execution_role

role = get_execution_role()

import mxnet as mx
data = mx.test_utils.get_mnist()

from mnist import train
model = train(data = data)

import os
os.mkdir('model')
model.save_checkpoint('model/model', 0000)
import tarfile
with tarfile.open('model.tar.gz', mode='w:gz') as archive:
    archive.add('model', recursive=True)

import sagemaker

sagemaker_session = sagemaker.Session()
inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')

from sagemaker.mxnet.model import MXNetModel
sagemaker_model = MXNetModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                  role = role,
                                  entry_point = 'mnist.py')

predictor = sagemaker_model.deploy(initial_instance_count=1,
                                          instance_type='ml.c4.xlarge')

predict_sample = data['test_data'][0][0]
response = predictor.predict(data)
print('Raw prediction result:')
print(response)

print(predictor.endpoint)

sagemaker.Session().delete_endpoint(predictor.endpoint)

os.remove('model.tar.gz')
import shutil
shutil.rmtree('export')

