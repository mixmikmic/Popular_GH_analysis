import os
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()

import utils
from tensorflow.contrib.learn.python.learn.datasets import mnist
import tensorflow as tf

data_sets = mnist.read_data_sets('data', dtype=tf.uint8, reshape=False, validation_size=5000)

utils.convert_to(data_sets.train, 'train', 'data')
utils.convert_to(data_sets.validation, 'validation', 'data')
utils.convert_to(data_sets.test, 'test', 'data')

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/mnist')

get_ipython().system("cat 'mnist.py'")

from sagemaker.tensorflow import TensorFlow

mnist_estimator = TensorFlow(entry_point='mnist.py',
                             role=role,
                             training_steps=1000, 
                             evaluation_steps=100,
                             train_instance_count=2,
                             train_instance_type='ml.c4.xlarge')

mnist_estimator.fit(inputs)

mnist_predictor = mnist_estimator.deploy(initial_instance_count=1,
                                             instance_type='ml.c4.xlarge')

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

for i in range(10):
    data = mnist.test.images[i].tolist()
    tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)
    predict_response = mnist_predictor.predict(tensor_proto)
    
    print("========================================")
    label = np.argmax(mnist.test.labels[i])
    print("label is {}".format(label))
    prediction = predict_response['outputs']['classes']['int64Val'][0]
    print("prediction is {}".format(prediction))

sagemaker.Session().delete_endpoint(mnist_predictor.endpoint)

