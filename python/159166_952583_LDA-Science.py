get_ipython().system('conda install -y scipy')

get_ipython().run_line_magic('matplotlib', 'inline')

import os, re, tarfile

import boto3
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# some helpful utility functions are defined in the Python module
# "generate_example_data" located in the same directory as this
# notebook
from generate_example_data import (
    generate_griffiths_data, match_estimated_topics,
    plot_lda, plot_lda_topics)

# accessing the SageMaker Python SDK
import sagemaker
from sagemaker.amazon.common import numpy_to_record_serializer
from sagemaker.predictor import csv_serializer, json_deserializer

from sagemaker import get_execution_role

role = get_execution_role()

bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/lda_science'


print('Training input/output will be stored in {}/{}'.format(bucket, prefix))
print('\nIAM Role: {}'.format(role))

print('Generating example data...')
num_documents = 6000
known_alpha, known_beta, documents, topic_mixtures = generate_griffiths_data(
    num_documents=num_documents, num_topics=10)
num_topics, vocabulary_size = known_beta.shape


# separate the generated data into training and tests subsets
num_documents_training = int(0.9*num_documents)
num_documents_test = num_documents - num_documents_training

documents_training = documents[:num_documents_training]
documents_test = documents[num_documents_training:]

topic_mixtures_training = topic_mixtures[:num_documents_training]
topic_mixtures_test = topic_mixtures[num_documents_training:]

print('documents_training.shape = {}'.format(documents_training.shape))
print('documents_test.shape = {}'.format(documents_test.shape))

print('First training document =\n{}'.format(documents_training[0]))
print('\nVocabulary size = {}'.format(vocabulary_size))
print('Length of first document = {}'.format(documents_training[0].sum()))

average_document_length = documents.sum(axis=1).mean()
print('Observed average document length = {}'.format(average_document_length))

print('First topic =\n{}'.format(known_beta[0]))

print('\nTopic-word probability matrix (beta) shape: (num_topics, vocabulary_size) = {}'.format(known_beta.shape))
print('\nSum of elements of first topic = {}'.format(known_beta[0].sum()))

print('Topic #1:\n{}'.format(known_beta[0]))
print('Topic #6:\n{}'.format(known_beta[5]))

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda(documents_training, nrows=3, ncols=4, cmap='gray_r', with_colorbar=True)
fig.suptitle('$w$ - Document Word Counts')
fig.set_dpi(160)

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda(known_beta, nrows=1, ncols=10)
fig.suptitle(r'Known $\beta$ - Topic-Word Probability Distributions')
fig.set_dpi(160)
fig.set_figheight(2)

print('First training document =\n{}'.format(documents_training[0]))
print('\nVocabulary size = {}'.format(vocabulary_size))
print('Length of first document = {}'.format(documents_training[0].sum()))

print('First training document topic mixture =\n{}'.format(topic_mixtures_training[0]))
print('\nNumber of topics = {}'.format(num_topics))
print('sum(theta) = {}'.format(topic_mixtures_training[0].sum()))

get_ipython().run_line_magic('matplotlib', 'inline')

fig, (ax1,ax2) = plt.subplots(2, 1)

ax1.matshow(documents[0].reshape(5,5), cmap='gray_r')
ax1.set_title(r'$w$ - Document', fontsize=20)
ax1.set_xticks([])
ax1.set_yticks([])

cax2 = ax2.matshow(topic_mixtures[0].reshape(1,-1), cmap='Reds', vmin=0, vmax=1)
cbar = fig.colorbar(cax2, orientation='horizontal')
ax2.set_title(r'$\theta$ - Topic Mixture', fontsize=20)
ax2.set_xticks([])
ax2.set_yticks([])

fig.set_dpi(100)

get_ipython().run_line_magic('matplotlib', 'inline')

# pot
fig = plot_lda(known_beta, nrows=1, ncols=10)
fig.suptitle(r'Known $\beta$ - Topic-Word Probability Distributions')
fig.set_dpi(160)
fig.set_figheight(1.5)

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda_topics(documents_training, 3, 4, topic_mixtures=topic_mixtures)
fig.suptitle(r'$(w,\theta)$ - Documents with Known Topic Mixtures')
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

# download and extract the model file from S3
job_name = lda.latest_training_job.job_name
model_fname = 'model.tar.gz'
model_object = os.path.join(prefix, 'output', job_name, 'output', model_fname)
boto3.Session().resource('s3').Bucket(bucket).Object(model_object).download_file(fname)
with tarfile.open(fname) as tar:
    tar.extractall()
print('Downloaded and extracted model tarball: {}'.format(model_object))

# obtain the model file
model_list = [fname for fname in os.listdir('.') if fname.startswith('model_')]
model_fname = model_list[0]
print('Found model file: {}'.format(model_fname))

# get the model from the model file and store in Numpy arrays
alpha, beta = mx.ndarray.load(model_fname)
learned_alpha_permuted = alpha.asnumpy()
learned_beta_permuted = beta.asnumpy()

print('\nLearned alpha.shape = {}'.format(learned_alpha_permuted.shape))
print('Learned beta.shape = {}'.format(learned_beta_permuted.shape))

permutation, learned_beta = match_estimated_topics(known_beta, learned_beta_permuted)
learned_alpha = learned_alpha_permuted[permutation]

fig = plot_lda(np.vstack([known_beta, learned_beta]), 2, 10)
fig.set_dpi(160)
fig.suptitle('Known vs. Found Topic-Word Probability Distributions')
fig.set_figheight(3)

beta_error = np.linalg.norm(known_beta - learned_beta, 1)
alpha_error = np.linalg.norm(known_alpha - learned_alpha, 1)
print('L1-error (beta) = {}'.format(beta_error))
print('L1-error (alpha) = {}'.format(alpha_error))

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

inferred_topic_mixtures_permuted = np.array([prediction['topic_mixture'] for prediction in results['predictions']])

print('Inferred topic mixtures (permuted):\n\n{}'.format(inferred_topic_mixtures_permuted))

inferred_topic_mixtures = inferred_topic_mixtures_permuted[:,permutation]

print('Inferred topic mixtures:\n\n{}'.format(inferred_topic_mixtures))

get_ipython().run_line_magic('matplotlib', 'inline')

# create array of bar plots
width = 0.4
x = np.arange(10)

nrows, ncols = 3, 4
fig, ax = plt.subplots(nrows, ncols, sharey=True)
for i in range(nrows):
    for j in range(ncols):
        index = i*ncols + j
        ax[i,j].bar(x, topic_mixtures_test[index], width, color='C0')
        ax[i,j].bar(x+width, inferred_topic_mixtures[index], width, color='C1')
        ax[i,j].set_xticks(range(num_topics))
        ax[i,j].set_yticks(np.linspace(0,1,5))
        ax[i,j].grid(which='major', axis='y')
        ax[i,j].set_ylim([0,1])
        ax[i,j].set_xticklabels([])
        if (i==(nrows-1)):
            ax[i,j].set_xticklabels(range(num_topics), fontsize=7)
        if (j==0):
            ax[i,j].set_yticklabels([0,'',0.5,'',1.0], fontsize=7)
        
fig.suptitle('Known vs. Inferred Topic Mixtures')
ax_super = fig.add_subplot(111, frameon=False)
ax_super.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax_super.grid(False)
ax_super.set_xlabel('Topic Index')
ax_super.set_ylabel('Topic Probability')
fig.set_dpi(160)

get_ipython().run_cell_magic('time', '', "\n# create a payload containing all of the test documents and run inference again\n#\n# TRY THIS:\n#   try switching between the test data set and a subset of the training\n#   data set. It is likely that LDA inference will perform better against\n#   the training set than the holdout test set.\n#\npayload_documents = documents_test                    # Example 1\nknown_topic_mixtures = topic_mixtures_test            # Example 1\n#payload_documents = documents_training[:600];         # Example 2\n#known_topic_mixtures = topic_mixtures_training[:600]  # Example 2\n\nprint('Invoking endpoint...\\n')\nresults = lda_inference.predict(payload_documents)\n\ninferred_topic_mixtures_permuted = np.array([prediction['topic_mixture'] for prediction in results['predictions']])\ninferred_topic_mixtures = inferred_topic_mixtures_permuted[:,permutation]\n\nprint('known_topics_mixtures.shape = {}'.format(known_topic_mixtures.shape))\nprint('inferred_topics_mixtures_test.shape = {}\\n'.format(inferred_topic_mixtures.shape))")

get_ipython().run_line_magic('matplotlib', 'inline')

l1_errors = np.linalg.norm((inferred_topic_mixtures - known_topic_mixtures), 1, axis=1)

# plot the error freqency
fig, ax_frequency = plt.subplots()
bins = np.linspace(0,1,40)
weights = np.ones_like(l1_errors)/len(l1_errors)
freq, bins, _ = ax_frequency.hist(l1_errors, bins=50, weights=weights, color='C0')
ax_frequency.set_xlabel('L1-Error')
ax_frequency.set_ylabel('Frequency', color='C0')


# plot the cumulative error
shift = (bins[1]-bins[0])/2
x = bins[1:] - shift
ax_cumulative = ax_frequency.twinx()
cumulative = np.cumsum(freq)/sum(freq)
ax_cumulative.plot(x, cumulative, marker='o', color='C1')
ax_cumulative.set_ylabel('Cumulative Frequency', color='C1')


# align grids and show
freq_ticks = np.linspace(0, 1.5*freq.max(), 5)
freq_ticklabels = np.round(100*freq_ticks)/100
ax_frequency.set_yticks(freq_ticks)
ax_frequency.set_yticklabels(freq_ticklabels)
ax_cumulative.set_yticks(np.linspace(0, 1, 5))
ax_cumulative.grid(which='major', axis='y')
ax_cumulative.set_ylim((0,1))


fig.suptitle('Topic Mixutre L1-Errors')
fig.set_dpi(110)

N = 6

good_idx = (l1_errors < 0.05)
good_documents = payload_documents[good_idx][:N]
good_topic_mixtures = inferred_topic_mixtures[good_idx][:N]

poor_idx = (l1_errors > 0.3)
poor_documents = payload_documents[poor_idx][:N]
poor_topic_mixtures = inferred_topic_mixtures[poor_idx][:N]

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda_topics(good_documents, 2, 3, topic_mixtures=good_topic_mixtures)
fig.suptitle('Documents With Accurate Inferred Topic-Mixtures')
fig.set_dpi(120)

get_ipython().run_line_magic('matplotlib', 'inline')

fig = plot_lda_topics(poor_documents, 2, 3, topic_mixtures=poor_topic_mixtures)
fig.suptitle('Documents With Inaccurate Inferred Topic-Mixtures')
fig.set_dpi(120)

sagemaker.Session().delete_endpoint(lda_inference.endpoint)



