from sagemaker import get_execution_role

role = get_execution_role()
bucket='<bucket-name>'

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nwith gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n    train_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (2,10)


def show_digit(img, caption='', subplot=None):
    if subplot==None:
        _,(subplot)=plt.subplots(1,1)
    imgr=img.reshape((28,28))
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)

show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30]))

get_ipython().run_cell_magic('time', '', "from sagemaker.amazon.common import write_numpy_to_dense_tensor\nimport io\nimport boto3\n\ndata_key = 'kmeans_lowlevel_example/data'\ndata_location = 's3://{}/{}'.format(bucket, data_key)\nprint('training data will be uploaded to: {}'.format(data_location))\n\n# Convert the training data into the format required by the SageMaker KMeans algorithm\nbuf = io.BytesIO()\nwrite_numpy_to_dense_tensor(buf, train_set[0], train_set[1])\nbuf.seek(0)\n\nboto3.resource('s3').Bucket(bucket).Object(data_key).upload_fileobj(buf)")

get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\njob_name = \'kmeans-lowlevel-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint("Training job", job_name)\n\nimages = {\'us-west-2\': \'174872318107.dkr.ecr.us-west-2.amazonaws.com/kmeans:latest\',\n          \'us-east-1\': \'382416733822.dkr.ecr.us-east-1.amazonaws.com/kmeans:latest\',\n          \'us-east-2\': \'404615174143.dkr.ecr.us-east-2.amazonaws.com/kmeans:latest\',\n          \'eu-west-1\': \'438346466558.dkr.ecr.eu-west-1.amazonaws.com/kmeans:latest\'}\nimage = images[boto3.Session().region_name]\n\noutput_location = \'s3://{}/kmeans_example/output\'.format(bucket)\nprint(\'training artifacts will be uploaded to: {}\'.format(output_location))\n\ncreate_training_params = \\\n{\n    "AlgorithmSpecification": {\n        "TrainingImage": image,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": output_location\n    },\n    "ResourceConfig": {\n        "InstanceCount": 2,\n        "InstanceType": "ml.c4.8xlarge",\n        "VolumeSizeInGB": 50\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "k": "10",\n        "feature_dim": "784",\n        "mini_batch_size": "500",\n        "force_dense": "True"\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 60 * 60\n    },\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": data_location,\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "CompressionType": "None",\n            "RecordWrapperType": "None"\n        }\n    ]\n}\n\n\nsagemaker = boto3.client(\'sagemaker\')\n\nsagemaker.create_training_job(**create_training_params)\n\nstatus = sagemaker.describe_training_job(TrainingJobName=job_name)[\'TrainingJobStatus\']\nprint(status)\n\ntry:\n    sagemaker.get_waiter(\'training_job_completed_or_stopped\').wait(TrainingJobName=job_name)\nfinally:\n    status = sagemaker.describe_training_job(TrainingJobName=job_name)[\'TrainingJobStatus\']\n    print("Training job ended with status: " + status)\n    if status == \'Failed\':\n        message = sagemaker.describe_training_job(TrainingJobName=job_name)[\'FailureReason\']\n        print(\'Training failed with the following error: {}\'.format(message))\n        raise Exception(\'Training job failed\')')

get_ipython().run_cell_magic('time', '', "import boto3\nfrom time import gmtime, strftime\n\n\nmodel_name=job_name\nprint(model_name)\n\ninfo = sagemaker.describe_training_job(TrainingJobName=job_name)\nmodel_data = info['ModelArtifacts']['S3ModelArtifacts']\n\nprimary_container = {\n    'Image': image,\n    'ModelDataUrl': model_data\n}\n\ncreate_model_response = sagemaker.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response['ModelArn'])")

from time import gmtime, strftime

endpoint_config_name = 'KMeansEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = sagemaker.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.c4.xlarge',
        'InitialInstanceCount':3,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'KMeansEndpoint-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = sagemaker.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sagemaker.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\ntry:\n    sagemaker.get_waiter(\'endpoint_in_service\').wait(EndpointName=endpoint_name)\nfinally:\n    resp = sagemaker.describe_endpoint(EndpointName=endpoint_name)\n    status = resp[\'EndpointStatus\']\n    print("Arn: " + resp[\'EndpointArn\'])\n    print("Create endpoint ended with status: " + status)\n\n    if status != \'InService\':\n        message = sagemaker.describe_endpoint(EndpointName=endpoint_name)[\'FailureReason\']\n        print(\'Training failed with the following error: {}\'.format(message))\n        raise Exception(\'Endpoint creation did not succeed\')')

# Simple function to create a csv from our numpy array
def np2csv(arr):
    csv = io.BytesIO()
    numpy.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()

runtime = boto3.Session().client('runtime.sagemaker')

import json

payload = np2csv(train_set[0][30:31])

response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='text/csv', 
                                   Body=payload)
result = json.loads(response['Body'].read().decode())
print(result)

get_ipython().run_cell_magic('time', '', '\npayload = np2csv(valid_set[0][0:100])\nresponse = runtime.invoke_endpoint(EndpointName=endpoint_name, \n                                   ContentType=\'text/csv\', \n                                   Body=payload)\nresult = json.loads(response[\'Body\'].read().decode())\nclusters = [p[\'closest_cluster\'] for p in result[\'predictions\']]\n\nfor cluster in range(10):\n    print(\'\\n\\n\\nCluster {}:\'.format(int(cluster)))\n    digits = [ img for l, img in zip(clusters, valid_set[0]) if int(l) == cluster ]\n    height=((len(digits)-1)//5)+1\n    width=5\n    plt.rcParams["figure.figsize"] = (width,height)\n    _, subplots = plt.subplots(height, width)\n    subplots=numpy.ndarray.flatten(subplots)\n    for subplot, image in zip(subplots, digits):\n        show_digit(image, subplot=subplot)\n    for subplot in subplots[len(digits):]:\n        subplot.axis(\'off\')\n\n    plt.show()')

sagemaker.delete_endpoint(EndpointName=endpoint_name)

