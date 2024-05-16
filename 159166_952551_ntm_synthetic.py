bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/ntm_synthetic'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

import numpy as np
from generate_example_data import generate_griffiths_data, plot_topic_data
import io
import os
import time
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import scipy
import sagemaker
import sagemaker.amazon.common as smac
from sagemaker.predictor import csv_serializer, json_deserializer

# generate the sample data
num_documents = 5000
num_topics = 5
vocabulary_size = 25
known_alpha, known_beta, documents, topic_mixtures = generate_griffiths_data(
    num_documents=num_documents, num_topics=num_topics, vocabulary_size=vocabulary_size)

# separate the generated data into training and tests subsets
num_documents_training = int(0.8*num_documents)
num_documents_test = num_documents - num_documents_training

documents_training = documents[:num_documents_training]
documents_test = documents[num_documents_training:]

topic_mixtures_training = topic_mixtures[:num_documents_training]
topic_mixtures_test = topic_mixtures[num_documents_training:]

data_training = (documents_training, np.zeros(num_documents_training))
data_test = (documents_test, np.zeros(num_documents_test))

print('First training document = {}'.format(documents[0]))
print('\nVocabulary size = {}'.format(vocabulary_size))

np.set_printoptions(precision=4, suppress=True)

print('Known topic mixture of first training document = {}'.format(topic_mixtures_training[0]))
print('\nNumber of topics = {}'.format(num_topics))

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_topic_data(documents_training[:10], nrows=2, ncols=5, cmap='gray_r', with_colorbar=False)
fig.suptitle('Example Documents')
fig.set_dpi(160)

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, data_training[0].astype('float32'))
buf.seek(0)

key = 'ntm.data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)

containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/ntm:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/ntm:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/ntm:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/ntm:latest'}

sess = sagemaker.Session()

ntm = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.c4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)
ntm.set_hyperparameters(num_topics=num_topics,
                        feature_dim=vocabulary_size)

ntm.fit({'train': s3_train_data})

ntm_predictor = ntm.deploy(initial_instance_count=1,
                           instance_type='ml.c4.xlarge')

ntm_predictor.content_type = 'text/csv'
ntm_predictor.serializer = csv_serializer
ntm_predictor.deserializer = json_deserializer

results = ntm_predictor.predict(documents_training[:10])
print(results)

predictions = np.array([prediction['topic_weights'] for prediction in results['predictions']])

print(predictions)

print(topic_mixtures_training[0])  # known topic mixture
print(predictions[0])  # computed topic mixture

def predict_batches(data, rows=1000):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = []
    for array in split_array:
        results = ntm_predictor.predict(array)
        predictions += [r['topic_weights'] for r in results['predictions']]
    return np.array(predictions)

predictions = predict_batches(documents_training)

data = pd.DataFrame(np.concatenate([topic_mixtures_training, predictions], axis=1), 
                    columns=['actual_{}'.format(i) for i in range(5)] + ['predictions_{}'.format(i) for i in range(5)])
display(data.corr())
pd.plotting.scatter_matrix(pd.DataFrame(np.concatenate([topic_mixtures_training, predictions], axis=1)), figsize=(12, 12))
plt.show()

sagemaker.Session().delete_endpoint(ntm_predictor.endpoint)

