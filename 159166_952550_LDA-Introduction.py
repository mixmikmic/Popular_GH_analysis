get_ipython().system('conda install -y scipy')

get_ipython().run_line_magic('matplotlib', 'inline')

import os, re

import boto3
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# some helpful utility functions are defined in the Python module
# "generate_example_data" located in the same directory as this
# notebook
from generate_example_data import generate_griffiths_data, plot_lda, match_estimated_topics

# accessing the SageMaker Python SDK
import sagemaker
from sagemaker.amazon.common import numpy_to_record_serializer
from sagemaker.predictor import csv_serializer, json_deserializer

from sagemaker import get_execution_role

role = get_execution_role()
bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/lda_introduction'

print('Training input/output will be stored in {}/{}'.format(bucket, prefix))
print('\nIAM Role: {}'.format(role))

print('Generating example data...')
num_documents = 6000
num_topics = 5
known_alpha, known_beta, documents, topic_mixtures = generate_griffiths_data(
    num_documents=num_documents, num_topics=num_topics)
vocabulary_size = len(documents[0])

# separate the generated data into training and tests subsets
num_documents_training = int(0.9*num_documents)
num_documents_test = num_documents - num_documents_training

documents_training = documents[:num_documents_training]
documents_test = documents[num_documents_training:]

topic_mixtures_training = topic_mixtures[:num_documents_training]
topic_mixtures_test = topic_mixtures[num_documents_training:]

print('documents_training.shape = {}'.format(documents_training.shape))
print('documents_test.shape = {}'.format(documents_test.shape))

print('First training document =\n{}'.format(documents[0]))
print('\nVocabulary size = {}'.format(vocabulary_size))

print('Known topic mixture of first document =\n{}'.format(topic_mixtures_training[0]))
print('\nNumber of topics = {}'.format(num_topics))
print('Sum of elements = {}'.format(topic_mixtures_training[0].sum()))

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda(documents_training, nrows=3, ncols=4, cmap='gray_r', with_colorbar=True)
fig.suptitle('Example Document Word Counts')
fig.set_dpi(160)

# convert documents_training to Protobuf RecordIO format
recordio_protobuf_serializer = numpy_to_record_serializer()
fbuffer = recordio_protobuf_serializer(documents_training)

# upload to S3 in bucket/prefix/train
fname = 'lda.data'
s3_object = os.path.join(prefix, 'train', fname)
boto3.Session().resource('s3').Bucket(bucket).Object(s3_object).upload_fileobj(fbuffer)

s3_train_data = 's3://{}/{}'.format(bucket, s3_object)
print('Uploaded data to S3: {}'.format(s3_train_data))

# select the algorithm container based on this notebook's current location
containers = {
    'us-west-2': '266724342769.dkr.ecr.us-west-2.amazonaws.com/lda:latest',
    'us-east-1': '766337827248.dkr.ecr.us-east-1.amazonaws.com/lda:latest',
    'us-east-2': '999911452149.dkr.ecr.us-east-2.amazonaws.com/lda:latest',
    'eu-west-1': '999678624901.dkr.ecr.eu-west-1.amazonaws.com/lda:latest'
}
region_name = boto3.Session().region_name
container = containers[region_name]

print('Using SageMaker LDA container: {} ({})'.format(container, region_name))

session = sagemaker.Session()

# specify general training job information
lda = sagemaker.estimator.Estimator(
    container,
    role,
    output_path='s3://{}/{}/output'.format(bucket, prefix),
    train_instance_count=1,
    train_instance_type='ml.c4.2xlarge',
    sagemaker_session=session,
)

# set algorithm-specific hyperparameters
lda.set_hyperparameters(
    num_topics=num_topics,
    feature_dim=vocabulary_size,
    mini_batch_size=num_documents_training,
    alpha0=1.0,
)

# run the training job on input data stored in S3
lda.fit({'train': s3_train_data})

print('Training job name: {}'.format(lda.latest_training_job.job_name))

lda_inference = lda.deploy(
    initial_instance_count=1,
    instance_type='ml.c4.xlarge',  # LDA inference works best on ml.c4 instances
)

print('Endpoint name: {}'.format(lda_inference.endpoint))

lda_inference.content_type = 'text/csv'
lda_inference.serializer = csv_serializer
lda_inference.deserializer = json_deserializer

results = lda_inference.predict(documents_test[:12])

print(results)

computed_topic_mixtures = np.array([prediction['topic_mixture'] for prediction in results['predictions']])

print(computed_topic_mixtures)

print(topic_mixtures_test[0])      # known test topic mixture
print(computed_topic_mixtures[0])  # computed topic mixture (topics permuted)

sagemaker.Session().delete_endpoint(lda_inference.endpoint)



