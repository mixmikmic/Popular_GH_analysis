import os
import boto3
import sagemaker
from sagemaker.mxnet import MXNet
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

role = get_execution_role()

get_ipython().run_cell_magic('bash', '', 'mkdir data\ncurl https://raw.githubusercontent.com/saurabh3949/Text-Classification-Datasets/master/stsa.binary.phrases.train > data/train\ncurl https://raw.githubusercontent.com/saurabh3949/Text-Classification-Datasets/master/stsa.binary.test > data/test ')

inputs = sagemaker_session.upload_data(path='data', key_prefix='data/sentiment')

get_ipython().system("cat 'sentiment.py'")

m = MXNet("sentiment.py", 
          role=role, 
          train_instance_count=1, 
          train_instance_type="ml.c4.2xlarge",
          hyperparameters={'batch_size': 8, 
                         'epochs': 2, 
                         'learning_rate': 0.01, 
                         'embedding_size': 50, 
                         'log_interval': 1000})

m.fit(inputs)

predictor = m.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')

data = ["this movie was extremely good .",
        "the plot was very boring .",
        "this film is so slick , superficial and trend-hoppy .",
        "i just could not watch it till the end .",
        "the movie was so enthralling !"]

response = predictor.predict(data)
print response

sagemaker.Session().delete_endpoint(predictor.endpoint)

