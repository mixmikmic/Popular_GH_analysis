import os
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/abalone')

get_ipython().system("cat 'abalone.py'")

from sagemaker.tensorflow import TensorFlow

abalone_estimator = TensorFlow(entry_point='abalone.py',
                               role=role,
                               training_steps= 100,                                  
                               evaluation_steps= 100,
                               hyperparameters={'learning_rate': 0.001},
                               train_instance_count=1,
                               train_instance_type='ml.c4.xlarge')

abalone_estimator.fit(inputs)

abalone_predictor = abalone_estimator.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')

import tensorflow as tf
import numpy as np

prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=os.path.join('data/abalone_predict.csv'), target_dtype=np.int, features_dtype=np.float32)

data = prediction_set.data[0]
tensor_proto = tf.make_tensor_proto(values=np.asarray(data), shape=[1, len(data)], dtype=tf.float32)

abalone_predictor.predict(tensor_proto)

sagemaker.Session().delete_endpoint(abalone_predictor.endpoint)

