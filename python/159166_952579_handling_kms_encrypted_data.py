get_ipython().run_cell_magic('time', '', "\nimport os\nimport io\nimport boto3\nimport pandas as pd\nimport numpy as np\nimport re\nfrom sagemaker import get_execution_role\n\nregion = boto3.Session().region_name\n\nrole = get_execution_role()\n\nkms_key_id = '<your-kms-key-id>'\n\nbucket='<s3-bucket>' # put your s3 bucket name here, and create s3 bucket\nprefix = 'sagemaker/kms'\n# customize to your bucket where you have stored the data\nbucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket)")

from sklearn.datasets import load_boston
boston = load_boston()
X = boston['data']
y = boston['target']
feature_names = boston['feature_names']
data = pd.DataFrame(X, columns=feature_names)
target = pd.DataFrame(y, columns={'MEDV'})
data['MEDV'] = y
local_file_name = 'boston.csv'
data.to_csv(local_file_name, header=False, index=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

def write_file(X, y, fname):
    feature_names = boston['feature_names']
    data = pd.DataFrame(X, columns=feature_names)
    target = pd.DataFrame(y, columns={'MEDV'})
    data['MEDV'] = y
    # bring this column to the front before writing the files
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    data.to_csv(fname, header=False, index=False)

train_file = 'train.csv'
validation_file = 'val.csv'
test_file = 'test.csv'
write_file(X_train, y_train, train_file)
write_file(X_val, y_val, validation_file)
write_file(X_test, y_test, test_file)

s3 = boto3.client('s3')

data_train = open(train_file, 'rb')
key_train = '{}/train/{}'.format(prefix,train_file)


print("Put object...")
s3.put_object(Bucket=bucket,
              Key=key_train,
              Body=data_train,
              ServerSideEncryption='aws:kms',
              SSEKMSKeyId=kms_key_id)
print("Done uploading the training dataset")

data_validation = open(validation_file, 'rb')
key_validation = '{}/validation/{}'.format(prefix,validation_file)

print("Put object...")
s3.put_object(Bucket=bucket,
              Key=key_validation,
              Body=data_validation,
              ServerSideEncryption='aws:kms',
              SSEKMSKeyId=kms_key_id)

print("Done uploading the validation dataset")

containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
container = containers[boto3.Session().region_name]

get_ipython().run_cell_magic('time', '', 'from time import gmtime, strftime\nimport time\n\njob_name = \'xgboost-single-regression\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint("Training job", job_name)\n\ncreate_training_params = \\\n{\n    "AlgorithmSpecification": {\n        "TrainingImage": container,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": bucket_path + "/"+ prefix + "/output"\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.m4.4xlarge",\n        "VolumeSizeInGB": 5\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "max_depth":"5",\n        "eta":"0.2",\n        "gamma":"4",\n        "min_child_weight":"6",\n        "subsample":"0.7",\n        "silent":"0",\n        "objective":"reg:linear",\n        "num_round":"5"\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 86400\n    },\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": bucket_path + "/"+ prefix + \'/train\',\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "csv",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": bucket_path + "/"+ prefix + \'/validation\',\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "csv",\n            "CompressionType": "None"\n        }\n    ]\n}\n\nclient = boto3.client(\'sagemaker\')\nclient.create_training_job(**create_training_params)\n\ntry:\n    # wait for the job to finish and report the ending status\n    client.get_waiter(\'training_job_completed_or_stopped\').wait(TrainingJobName=job_name)\n    training_info = client.describe_training_job(TrainingJobName=job_name)\n    status = training_info[\'TrainingJobStatus\']\n    print("Training job ended with status: " + status)\nexcept:\n    print(\'Training failed to start\')\n     # if exception is raised, that means it has failed\n    message = client.describe_training_job(TrainingJobName=job_name)[\'FailureReason\']\n    print(\'Training failed with the following error: {}\'.format(message))')

get_ipython().run_cell_magic('time', '', "import boto3\nfrom time import gmtime, strftime\n\nmodel_name=job_name + '-model'\nprint(model_name)\n\ninfo = client.describe_training_job(TrainingJobName=job_name)\nmodel_data = info['ModelArtifacts']['S3ModelArtifacts']\nprint(model_data)\n\nprimary_container = {\n    'Image': container,\n    'ModelDataUrl': model_data\n}\n\ncreate_model_response = client.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response['ModelArn'])")

from time import gmtime, strftime

endpoint_config_name = 'XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = client.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.m4.xlarge',
        'InitialVariantWeight':1,
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'XGBoostEndpoint-new-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = client.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\n\nprint(\'EndpointArn = {}\'.format(create_endpoint_response[\'EndpointArn\']))\n\n# get the status of the endpoint\nresponse = client.describe_endpoint(EndpointName=endpoint_name)\nstatus = response[\'EndpointStatus\']\nprint(\'EndpointStatus = {}\'.format(status))\n\n\n# wait until the status has changed\nclient.get_waiter(\'endpoint_in_service\').wait(EndpointName=endpoint_name)\n\n\n# print the status of the endpoint\nendpoint_response = client.describe_endpoint(EndpointName=endpoint_name)\nstatus = endpoint_response[\'EndpointStatus\']\nprint(\'Endpoint creation ended with EndpointStatus = {}\'.format(status))\n\nif status != \'InService\':\n    raise Exception(\'Endpoint creation failed.\')')

runtime_client = boto3.client('runtime.sagemaker')

import sys
import math
def do_predict(data, endpoint_name, content_type):
    payload = ''.join(data)
    response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType=content_type, 
                                   Body=payload)
    result = response['Body'].read()
    result = result.decode("utf-8")
    result = result.split(',')
    return result

def batch_predict(data, batch_size, endpoint_name, content_type):
    items = len(data)
    arrs = []
    
    for offset in range(0, items, batch_size):
        if offset+batch_size < items:
            results = do_predict(data[offset:(offset+batch_size)], endpoint_name, content_type)
            arrs.extend(results)
        else:
            arrs.extend(do_predict(data[offset:items], endpoint_name, content_type))
        sys.stdout.write('.')
    return(arrs)

get_ipython().run_cell_magic('time', '', "import json\nimport numpy as np\n\n\nwith open('test.csv') as f:\n    lines = f.readlines()\n\n#remove the labels\nlabels = [line.split(',')[0] for line in lines]\nfeatures = [line.split(',')[1:] for line in lines]\n\nfeatures_str = [','.join(row) for row in features]\npreds = batch_predict(features_str, 100, endpoint_name, 'text/csv')\nprint('\\n Median Absolute Percent Error (MdAPE) = ', np.median(np.abs(np.asarray(labels, dtype=float) - np.asarray(preds, dtype=float)) / np.asarray(labels, dtype=float)))")

client.delete_endpoint(EndpointName=endpoint_name)

