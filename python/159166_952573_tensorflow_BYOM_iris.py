import boto3, re
from sagemaker import get_execution_role

role = get_execution_role()

get_ipython().system('cat iris_dnn_classifier.py')

from iris_dnn_classifier import estimator_fn
classifier = estimator_fn(run_config = None, params = None)

import os 
from six.moves.urllib.request import urlopen

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
      f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
      f.write(raw)

from iris_dnn_classifier import train_input_fn
train_func = train_input_fn('.', params = None)

classifier.train(input_fn = train_func, steps = 1000)

from iris_dnn_classifier import serving_input_fn

exported_model = classifier.export_savedmodel(export_dir_base = 'export/Servo/', 
                               serving_input_receiver_fn = serving_input_fn)

print (exported_model)
import tarfile
with tarfile.open('model.tar.gz', mode='w:gz') as archive:
    archive.add('export', recursive=True)

import sagemaker

sagemaker_session = sagemaker.Session()
inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')

from sagemaker.tensorflow.model import TensorFlowModel
sagemaker_model = TensorFlowModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                  role = role,
                                  entry_point = 'iris_dnn_classifier.py')

get_ipython().run_cell_magic('time', '', "predictor = sagemaker_model.deploy(initial_instance_count=1,\n                                          instance_type='ml.c4.xlarge')")

sample = [6.4,3.2,4.5,1.5]
predictor.predict(sample)

os.remove('model.tar.gz')
import shutil
shutil.rmtree('export')

sagemaker.Session().delete_endpoint(predictor.endpoint)

