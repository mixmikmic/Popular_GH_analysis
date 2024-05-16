get_ipython().run_cell_magic('time', '', "\nimport os\nimport boto3\nimport re\nimport copy\nimport time\nfrom time import gmtime, strftime\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\n\nregion = boto3.Session().region_name\n\nbucket='<bucket-name>' # put your s3 bucket name here, and create s3 bucket\nprefix = 'sagemaker/xgboost-multiclass-classification'\n# customize to your bucket where you have stored the data\nbucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket)")

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nf = gzip.open(\'mnist.pkl.gz\', \'rb\')\ntrain_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')\nf.close()')

get_ipython().run_cell_magic('time', '', '\nimport struct\nimport io\nimport boto3\n\n \ndef to_libsvm(f, labels, values):\n     f.write(bytes(\'\\n\'.join(\n         [\'{} {}\'.format(label, \' \'.join([\'{}:{}\'.format(i + 1, el) for i, el in enumerate(vec)])) for label, vec in\n          zip(labels, values)]), \'utf-8\'))\n     return f\n\n\ndef write_to_s3(fobj, bucket, key):\n    return boto3.Session().resource(\'s3\').Bucket(bucket).Object(key).upload_fileobj(fobj)\n\ndef get_dataset():\n  import pickle\n  import gzip\n  with gzip.open(\'mnist.pkl.gz\', \'rb\') as f:\n      u = pickle._Unpickler(f)\n      u.encoding = \'latin1\'\n      return u.load()\n\ndef upload_to_s3(partition_name, partition):\n    labels = [t.tolist() for t in partition[1]]\n    vectors = [t.tolist() for t in partition[0]]\n    num_partition = 5                                 # partition file into 5 parts\n    partition_bound = int(len(labels)/num_partition)\n    for i in range(num_partition):\n        f = io.BytesIO()\n        to_libsvm(f, labels[i*partition_bound:(i+1)*partition_bound], vectors[i*partition_bound:(i+1)*partition_bound])\n        f.seek(0)\n        key = "{}/{}/examples{}".format(prefix,partition_name,str(i))\n        url = \'s3n://{}/{}\'.format(bucket, key)\n        print(\'Writing to {}\'.format(url))\n        write_to_s3(f, bucket, key)\n        print(\'Done writing to {}\'.format(url))\n\ndef download_from_s3(partition_name, number, filename):\n    key = "{}/{}/examples{}".format(prefix,partition_name, number)\n    url = \'s3n://{}/{}\'.format(bucket, key)\n    print(\'Reading from {}\'.format(url))\n    s3 = boto3.resource(\'s3\')\n    s3.Bucket(bucket).download_file(key, filename)\n    try:\n        s3.Bucket(bucket).download_file(key, \'mnist.local.test\')\n    except botocore.exceptions.ClientError as e:\n        if e.response[\'Error\'][\'Code\'] == "404":\n            print(\'The object does not exist at {}.\'.format(url))\n        else:\n            raise        \n        \ndef convert_data():\n    train_set, valid_set, test_set = get_dataset()\n    partitions = [(\'train\', train_set), (\'validation\', valid_set), (\'test\', test_set)]\n    for partition_name, partition in partitions:\n        print(\'{}: {} {}\'.format(partition_name, partition[0].shape, partition[1].shape))\n        upload_to_s3(partition_name, partition)')

get_ipython().run_cell_magic('time', '', '\nconvert_data()')

containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
container = containers[boto3.Session().region_name]

#Ensure that the train and validation data folders generated above are reflected in the "InputDataConfig" parameter below.
common_training_params = {
    "AlgorithmSpecification": {
        "TrainingImage": container,
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": bucket_path + "/"+ prefix + "/xgboost"
    },
    "ResourceConfig": {
        "InstanceCount": 1,   
        "InstanceType": "ml.m4.10xlarge",
        "VolumeSizeInGB": 5
    },
    "HyperParameters": {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "silent":"0",
        "objective": "multi:softmax",
        "num_class": "10",
        "num_round": "10"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 86400
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": bucket_path + "/"+ prefix+ '/train/',
                    "S3DataDistributionType": "FullyReplicated" 
                }
            },
            "ContentType": "libsvm",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": bucket_path + "/"+ prefix+ '/validation/',
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "libsvm",
            "CompressionType": "None"
        }
    ]
}

#single machine job params
single_machine_job_name = 'xgboost-single-machine-classification' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Job name is:", single_machine_job_name)

single_machine_job_params = copy.deepcopy(common_training_params)
single_machine_job_params['TrainingJobName'] = single_machine_job_name
single_machine_job_params['OutputDataConfig']['S3OutputPath'] = bucket_path + "/"+ prefix + "/xgboost-single"
single_machine_job_params['ResourceConfig']['InstanceCount'] = 1

#distributed job params
distributed_job_name = 'xgboost-distributed-classification' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Job name is:", distributed_job_name)

distributed_job_params = copy.deepcopy(common_training_params)
distributed_job_params['TrainingJobName'] = distributed_job_name
distributed_job_params['OutputDataConfig']['S3OutputPath'] = bucket_path + "/"+ prefix + "/xgboost-distributed"
#number of instances used for training
distributed_job_params['ResourceConfig']['InstanceCount'] = 2 # no more than 5 if there are total 5 partition files generated above

# data distribution type for train channel
distributed_job_params['InputDataConfig'][0]['DataSource']['S3DataSource']['S3DataDistributionType'] = 'ShardedByS3Key'
# data distribution type for validation channel
distributed_job_params['InputDataConfig'][1]['DataSource']['S3DataSource']['S3DataDistributionType'] = 'ShardedByS3Key'

get_ipython().run_cell_magic('time', '', '\nregion = boto3.Session().region_name\nsm = boto3.Session().client(\'sagemaker\')\n\nsm.create_training_job(**single_machine_job_params)\nsm.create_training_job(**distributed_job_params)\n\nstatus = sm.describe_training_job(TrainingJobName=distributed_job_name)[\'TrainingJobStatus\']\nprint(status)\nsm.get_waiter(\'training_job_completed_or_stopped\').wait(TrainingJobName=distributed_job_name)\nstatus = sm.describe_training_job(TrainingJobName=distributed_job_name)[\'TrainingJobStatus\']\nprint("Training job ended with status: " + status)\nif status == \'Failed\':\n    message = sm.describe_training_job(TrainingJobName=distributed_job_name)[\'FailureReason\']\n    print(\'Training failed with the following error: {}\'.format(message))\n    raise Exception(\'Training job failed\')')

print('Single Machine:', sm.describe_training_job(TrainingJobName=single_machine_job_name)['TrainingJobStatus'])
print('Distributed:', sm.describe_training_job(TrainingJobName=distributed_job_name)['TrainingJobStatus'])

get_ipython().run_cell_magic('time', '', "import boto3\nfrom time import gmtime, strftime\n\nmodel_name=distributed_job_name + '-model'\nprint(model_name)\n\ninfo = sm.describe_training_job(TrainingJobName=distributed_job_name)\nmodel_data = info['ModelArtifacts']['S3ModelArtifacts']\nprint(model_data)\n\nprimary_container = {\n    'Image': container,\n    'ModelDataUrl': model_data\n}\n\ncreate_model_response = sm.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response['ModelArn'])")

from time import gmtime, strftime

endpoint_config_name = 'XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.c4.xlarge',
        'InitialVariantWeight':1,
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'XGBoostEndpoint-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = sm.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sm.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\nwhile status==\'Creating\':\n    time.sleep(60)\n    resp = sm.describe_endpoint(EndpointName=endpoint_name)\n    status = resp[\'EndpointStatus\']\n    print("Status: " + status)\n\nprint("Arn: " + resp[\'EndpointArn\'])\nprint("Status: " + status)')

runtime_client = boto3.client('runtime.sagemaker')

download_from_s3('test', 0, 'mnist.local.test') # reading the first part file within test

get_ipython().system('head -1 mnist.local.test > mnist.single.test')

get_ipython().run_cell_magic('time', '', "import json\n\nfile_name = 'mnist.single.test' #customize to your test file 'mnist.single.test' if use the data above\n\nwith open(file_name, 'r') as f:\n    payload = f.read()\n\nresponse = runtime_client.invoke_endpoint(EndpointName=endpoint_name, \n                                   ContentType='text/x-libsvm', \n                                   Body=payload)\nresult = response['Body'].read().decode('ascii')\nprint('Predicted label is {}.'.format(result))")

import sys
def do_predict(data, endpoint_name, content_type):
    payload = '\n'.join(data)
    response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType=content_type, 
                                   Body=payload)
    result = response['Body'].read().decode('ascii')
    preds = [float(num) for num in result.split(',')]
    return preds

def batch_predict(data, batch_size, endpoint_name, content_type):
    items = len(data)
    arrs = []
    for offset in range(0, items, batch_size):
        arrs.extend(do_predict(data[offset:min(offset+batch_size, items)], endpoint_name, content_type))
        sys.stdout.write('.')
    return(arrs)

get_ipython().run_cell_magic('time', '', "import json\n\nfile_name = 'mnist.local.test'\nwith open(file_name, 'r') as f:\n    payload = f.read().strip()\n\nlabels = [float(line.split(' ')[0]) for line in payload.split('\\n')]\ntest_data = payload.split('\\n')\npreds = batch_predict(test_data, 100, endpoint_name, 'text/x-libsvm')\n\nprint ('\\nerror rate=%f' % ( sum(1 for i in range(len(preds)) if preds[i]!=labels[i]) /float(len(preds))))")

preds[0:10]

labels[0:10]

import numpy
def error_rate(predictions, labels):
    """Return the error rate and confusions."""
    correct = numpy.sum(predictions == labels)
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = numpy.zeros([10, 10], numpy.int32)
    bundled = zip(predictions, labels)
    for predicted, actual in bundled:
        confusions[int(predicted), int(actual)] += 1
    
    return error, confusions

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

NUM_LABELS = 10  # change it according to num_class in your dataset
test_error, confusions = error_rate(numpy.asarray(preds), numpy.asarray(labels))
print('Test error: %.1f%%' % test_error)

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(False)
plt.xticks(numpy.arange(NUM_LABELS))
plt.yticks(numpy.arange(NUM_LABELS))
plt.imshow(confusions, cmap=plt.cm.jet, interpolation='nearest');

for i, cas in enumerate(confusions):
    for j, count in enumerate(cas):
        if count > 0:
            xoff = .07 * len(str(count))
            plt.text(j-xoff, i+.2, int(count), fontsize=9, color='white')

sm.delete_endpoint(EndpointName=endpoint_name)

