from sagemaker import get_execution_role

#Bucket location to save your custom code in tar.gz format.
custom_code_upload_location = 's3://<bucket-name>/customcode/mxnet'

#Bucket location where results of model training are saved.
model_artifacts_location = 's3://<bucket-name>/artifacts'

#IAM execution role that gives SageMaker access to resources in your AWS account.
#We can use the SageMaker Python SDK to get the role from our notebook environment. 
role = get_execution_role()

get_ipython().system('cat mnist.py')

from sagemaker.mxnet import MXNet

mnist_estimator = MXNet(entry_point='mnist.py',
                        role=role,
                        output_path=model_artifacts_location,
                        code_location=custom_code_upload_location,
                        train_instance_count=1, 
                        train_instance_type='ml.m4.xlarge',
                        hyperparameters={'learning_rate': 0.1})

get_ipython().run_cell_magic('time', '', "import boto3\n\nregion = boto3.Session().region_name\ntrain_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/train'.format(region)\ntest_data_location = 's3://sagemaker-sample-data-{}/mxnet/mnist/test'.format(region)\n\nmnist_estimator.fit({'train': train_data_location, 'test': test_data_location})")

get_ipython().run_cell_magic('time', '', "\npredictor = mnist_estimator.deploy(initial_instance_count=1,\n                                   instance_type='ml.c4.xlarge')")

from IPython.display import HTML
HTML(open("input.html").read())

response = predictor.predict(data)
print('Raw prediction result:')
print(response)

labeled_predictions = list(zip(range(10), response[0]))
print('Labeled predictions: ')
print(labeled_predictions)

labeled_predictions.sort(key=lambda label_and_prob: 1.0 - label_and_prob[1])
print('Most likely answer: {}'.format(labeled_predictions[0]))

print("Endpoint name: " + predictor.endpoint)

import sagemaker

sagemaker.Session().delete_endpoint(predictor.endpoint)



