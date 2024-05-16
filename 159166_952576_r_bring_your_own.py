bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/r_byo'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

import time
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_cell_magic('sh', '', '\n# The name of our algorithm\nalgorithm_name=rmars\n\n#set -e # stop if anything fails\n\naccount=$(aws sts get-caller-identity --query Account --output text)\n\n# Get the region defined in the current configuration (default to us-west-2 if none defined)\nregion=$(aws configure get region)\nregion=${region:-us-west-2}\n\nfullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"\n\n# If the repository doesn\'t exist in ECR, create it.\n\naws ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1\n\nif [ $? -ne 0 ]\nthen\n    aws ecr create-repository --repository-name "${algorithm_name}" > /dev/null\nfi\n\n# Get the login command from ECR and execute it directly\n$(aws ecr get-login --region ${region} --no-include-email)\n\n# On a SageMaker Notebook Instance, the docker daemon may need to be restarted in order\n# to detect your network configuration correctly.  (This is a known issue.)\nif [ -d "/home/ec2-user/SageMaker" ]; then\n  sudo service docker restart\nfi\n\n# Build the docker image locally with the image name and then push it to ECR\n# with the full name.\ndocker build  -t ${algorithm_name} .\ndocker tag ${algorithm_name} ${fullname}\n\ndocker push ${fullname}')

train_file = 'iris.csv'
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_file(train_file)

region = boto3.Session().region_name
account = boto3.client('sts').get_caller_identity().get('Account')

r_job = 'r-byo-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

print("Training job", r_job)

r_training_params = {
    "RoleArn": role,
    "TrainingJobName": r_job,
    "AlgorithmSpecification": {
        "TrainingImage": '{}.dkr.ecr.{}.amazonaws.com/rmars:latest'.format(account, region),
        "TrainingInputMode": "File"
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.m4.xlarge",
        "VolumeSizeInGB": 10
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        }
    ],
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}/output".format(bucket, prefix)
    },
    "HyperParameters": {
        "target": "Sepal.Length",
        "degree": "2"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    }
}

get_ipython().run_cell_magic('time', '', '\nsm = boto3.client(\'sagemaker\')\nsm.create_training_job(**r_training_params)\n\nstatus = sm.describe_training_job(TrainingJobName=r_job)[\'TrainingJobStatus\']\nprint(status)\nsm.get_waiter(\'training_job_completed_or_stopped\').wait(TrainingJobName=r_job)\nstatus = sm.describe_training_job(TrainingJobName=r_job)[\'TrainingJobStatus\']\nprint("Training job ended with status: " + status)\nif status == \'Failed\':\n    message = sm.describe_training_job(TrainingJobName=r_job)[\'FailureReason\']\n    print(\'Training failed with the following error: {}\'.format(message))\n    raise Exception(\'Training job failed\')')

r_hosting_container = {
    'Image': '{}.dkr.ecr.{}.amazonaws.com/rmars:latest'.format(account, region),
    'ModelDataUrl': sm.describe_training_job(TrainingJobName=r_job)['ModelArtifacts']['S3ModelArtifacts']
}

create_model_response = sm.create_model(
    ModelName=r_job,
    ExecutionRoleArn=role,
    PrimaryContainer=r_hosting_container)

print(create_model_response['ModelArn'])

r_endpoint_config = 'r-endpoint-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print(r_endpoint_config)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=r_endpoint_config,
    ProductionVariants=[{
        'InstanceType': 'ml.m4.xlarge',
        'InitialInstanceCount': 1,
        'ModelName': r_job,
        'VariantName': 'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

get_ipython().run_cell_magic('time', '', '\nr_endpoint = \'r-endpoint-\' + time.strftime("%Y%m%d%H%M", time.gmtime())\nprint(r_endpoint)\ncreate_endpoint_response = sm.create_endpoint(\n    EndpointName=r_endpoint,\n    EndpointConfigName=r_endpoint_config)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sm.describe_endpoint(EndpointName=r_endpoint)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\ntry:\n    sm.get_waiter(\'endpoint_in_service\').wait(EndpointName=r_endpoint)\nfinally:\n    resp = sm.describe_endpoint(EndpointName=r_endpoint)\n    status = resp[\'EndpointStatus\']\n    print("Arn: " + resp[\'EndpointArn\'])\n    print("Status: " + status)\n\n    if status != \'InService\':\n        raise Exception(\'Endpoint creation did not succeed\')')

iris = pd.read_csv('iris.csv')

runtime = boto3.Session().client('runtime.sagemaker')

payload = iris.drop(['Sepal.Length'], axis=1).to_csv(index=False)
response = runtime.invoke_endpoint(EndpointName=r_endpoint,
                                   ContentType='text/csv',
                                   Body=payload)

result = json.loads(response['Body'].read().decode())
result 

plt.scatter(iris['Sepal.Length'], np.fromstring(result[0], sep=','))
plt.show()

sm.delete_endpoint(EndpointName=r_endpoint)

