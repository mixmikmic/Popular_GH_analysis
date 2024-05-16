get_ipython().run_cell_magic('time', '', "\nimport os\nimport boto3\nimport re\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\nregion = boto3.Session().region_name\n\nbucket='<bucket-name>' # put your s3 bucket name here, and create s3 bucket\nprefix = 'sagemaker/xgboost-regression'\n# customize to your bucket where you have stored the data\nbucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket)")

get_ipython().run_cell_magic('time', '', "\nimport io\nimport boto3\nimport random\n\ndef data_split(FILE_DATA, FILE_TRAIN, FILE_VALIDATION, FILE_TEST, PERCENT_TRAIN, PERCENT_VALIDATION, PERCENT_TEST):\n    data = [l for l in open(FILE_DATA, 'r')]\n    train_file = open(FILE_TRAIN, 'w')\n    valid_file = open(FILE_VALIDATION, 'w')\n    tests_file = open(FILE_TEST, 'w')\n\n    num_of_data = len(data)\n    num_train = int((PERCENT_TRAIN/100.0)*num_of_data)\n    num_valid = int((PERCENT_VALIDATION/100.0)*num_of_data)\n    num_tests = int((PERCENT_TEST/100.0)*num_of_data)\n\n    data_fractions = [num_train, num_valid, num_tests]\n    split_data = [[],[],[]]\n\n    rand_data_ind = 0\n\n    for split_ind, fraction in enumerate(data_fractions):\n        for i in range(fraction):\n            rand_data_ind = random.randint(0, len(data)-1)\n            split_data[split_ind].append(data[rand_data_ind])\n            data.pop(rand_data_ind)\n\n    for l in split_data[0]:\n        train_file.write(l)\n\n    for l in split_data[1]:\n        valid_file.write(l)\n\n    for l in split_data[2]:\n        tests_file.write(l)\n\n    train_file.close()\n    valid_file.close()\n    tests_file.close()\n\ndef write_to_s3(fobj, bucket, key):\n    return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(fobj)\n\ndef upload_to_s3(bucket, channel, filename):\n    fobj=open(filename, 'rb')\n    key = prefix+'/'+channel\n    url = 's3://{}/{}/{}'.format(bucket, key, filename)\n    print('Writing to {}'.format(url))\n    write_to_s3(fobj, bucket, key)")

get_ipython().run_cell_magic('time', '', 'import urllib.request\n\n# Load the dataset\nFILE_DATA = \'abalone\'\nurllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/abalone", FILE_DATA)\n\n#split the downloaded data into train/test/validation files\nFILE_TRAIN = \'abalone.train\'\nFILE_VALIDATION = \'abalone.validation\'\nFILE_TEST = \'abalone.test\'\nPERCENT_TRAIN = 70\nPERCENT_VALIDATION = 15\nPERCENT_TEST = 15\ndata_split(FILE_DATA, FILE_TRAIN, FILE_VALIDATION, FILE_TEST, PERCENT_TRAIN, PERCENT_VALIDATION, PERCENT_TEST)\n\n#upload the files to the S3 bucket\nupload_to_s3(bucket, \'train\', FILE_TRAIN)\nupload_to_s3(bucket, \'validation\', FILE_VALIDATION)\nupload_to_s3(bucket, \'test\', FILE_TEST)')

containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
container = containers[boto3.Session().region_name]

get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\njob_name = \'xgboost-single-machine-regression-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint("Training job", job_name)\n\n#Ensure that the training and validation data folders generated above are reflected in the "InputDataConfig" parameter below.\n\ncreate_training_params = \\\n{\n    "AlgorithmSpecification": {\n        "TrainingImage": container,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": bucket_path + "/" + prefix + "/single-xgboost"\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.m4.4xlarge",\n        "VolumeSizeInGB": 5\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "max_depth":"5",\n        "eta":"0.2",\n        "gamma":"4",\n        "min_child_weight":"6",\n        "subsample":"0.7",\n        "silent":"0",\n        "objective":"reg:linear",\n        "num_round":"50"\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 3600\n    },\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": bucket_path + "/" + prefix + \'/train\',\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "libsvm",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": bucket_path + "/" + prefix + \'/validation\',\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "libsvm",\n            "CompressionType": "None"\n        }\n    ]\n}\n\n\nclient = boto3.client(\'sagemaker\')\nclient.create_training_job(**create_training_params)\n\nimport time\n\nstatus = client.describe_training_job(TrainingJobName=job_name)[\'TrainingJobStatus\']\nprint(status)\nwhile status !=\'Completed\' and status!=\'Failed\':\n    time.sleep(60)\n    status = client.describe_training_job(TrainingJobName=job_name)[\'TrainingJobStatus\']\n    print(status)')

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

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'XGBoostEndpoint-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = client.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = client.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\nwhile status==\'Creating\':\n    time.sleep(60)\n    resp = client.describe_endpoint(EndpointName=endpoint_name)\n    status = resp[\'EndpointStatus\']\n    print("Status: " + status)\n\nprint("Arn: " + resp[\'EndpointArn\'])\nprint("Status: " + status)')

runtime_client = boto3.client('runtime.sagemaker')

get_ipython().system('head -1 abalone.test > abalone.single.test')

get_ipython().run_cell_magic('time', '', 'import json\nfrom itertools import islice\nimport math\nimport struct\n\nfile_name = \'abalone.single.test\' #customize to your test file\nwith open(file_name, \'r\') as f:\n    payload = f.read().strip()\nresponse = runtime_client.invoke_endpoint(EndpointName=endpoint_name, \n                                   ContentType=\'text/x-libsvm\', \n                                   Body=payload)\nresult = response[\'Body\'].read()\nresult = result.decode("utf-8")\nresult = result.split(\',\')\nresult = [math.ceil(float(i)) for i in result]\nlabel = payload.strip(\' \')[0]\nprint (\'Label: \',label,\'\\nPrediction: \', result[0])')

import sys
import math
def do_predict(data, endpoint_name, content_type):
    payload = '\n'.join(data)
    response = runtime_client.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType=content_type, 
                                   Body=payload)
    result = response['Body'].read()
    result = result.decode("utf-8")
    result = result.split(',')
    preds = [float((num)) for num in result]
    preds = [math.ceil(num) for num in preds]
    return preds

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

get_ipython().run_cell_magic('time', '', "import json\nimport numpy as np\n\nwith open(FILE_TEST, 'r') as f:\n    payload = f.read().strip()\n\nlabels = [int(line.split(' ')[0]) for line in payload.split('\\n')]\ntest_data = [line for line in payload.split('\\n')]\npreds = batch_predict(test_data, 100, endpoint_name, 'text/x-libsvm')\n\nprint('\\n Median Absolute Percent Error (MdAPE) = ', np.median(np.abs(np.array(labels) - np.array(preds)) / np.array(labels)))")

client.delete_endpoint(EndpointName=endpoint_name)

