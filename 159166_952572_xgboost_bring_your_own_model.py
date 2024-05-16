get_ipython().run_cell_magic('time', '', "\nimport os\nimport boto3\nimport re\nimport json\nfrom sagemaker import get_execution_role\n\nregion = boto3.Session().region_name\n\nrole = get_execution_role()\n\nbucket='<s3 bucket>' # put your s3 bucket name here, and create s3 bucket\nprefix = 'sagemaker/xgboost-byo'\nbucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket)\n# customize to your bucket where you have stored the data")

get_ipython().system('conda install -y -c conda-forge xgboost')

get_ipython().run_cell_magic('time', '', 'import pickle, gzip, numpy, urllib.request, json\n\n# Load the dataset\nurllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")\nf = gzip.open(\'mnist.pkl.gz\', \'rb\')\ntrain_set, valid_set, test_set = pickle.load(f, encoding=\'latin1\')\nf.close()')

get_ipython().run_cell_magic('time', '', "\nimport struct\nimport io\nimport boto3\n\ndef get_dataset():\n  import pickle\n  import gzip\n  with gzip.open('mnist.pkl.gz', 'rb') as f:\n      u = pickle._Unpickler(f)\n      u.encoding = 'latin1'\n      return u.load()")

train_set, valid_set, test_set = get_dataset()

train_X = train_set[0]
train_y = train_set[1]

valid_X = valid_set[0]
valid_y = valid_set[1]

test_X = test_set[0]
test_y = test_set[1]

import xgboost as xgb
import sklearn as sk 

bt = xgb.XGBClassifier(max_depth=5,
                       learning_rate=0.2,
                       n_estimators=10,
                       objective='multi:softmax')   # Setup xgboost model
bt.fit(train_X, train_y, # Train it to our data
       eval_set=[(valid_X, valid_y)], 
       verbose=False)

model_file_name = "locally-trained-xgboost-model"
bt._Booster.save_model(model_file_name)

get_ipython().system('tar czvf model.tar.gz $model_file_name')

fObj = open("model.tar.gz", 'rb')
key= os.path.join(prefix, model_file_name, 'model.tar.gz')
boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(fObj)

containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}
container = containers[boto3.Session().region_name]

get_ipython().run_cell_magic('time', '', 'from time import gmtime, strftime\n\nmodel_name = model_file_name + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nmodel_url = \'https://s3-{}.amazonaws.com/{}/{}\'.format(region,bucket,key)\nsm_client = boto3.client(\'sagemaker\')\n\nprint (model_url)\n\nprimary_container = {\n    \'Image\': container,\n    \'ModelDataUrl\': model_url,\n}\n\ncreate_model_response2 = sm_client.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response2[\'ModelArn\'])')

from time import gmtime, strftime

endpoint_config_name = 'XGBoostEndpointConfig-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print(endpoint_config_name)
create_endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.m4.xlarge',
        'InitialInstanceCount':1,
        'InitialVariantWeight':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

get_ipython().run_cell_magic('time', '', 'import time\n\nendpoint_name = \'XGBoostEndpoint-\' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())\nprint(endpoint_name)\ncreate_endpoint_response = sm_client.create_endpoint(\n    EndpointName=endpoint_name,\n    EndpointConfigName=endpoint_config_name)\nprint(create_endpoint_response[\'EndpointArn\'])\n\nresp = sm_client.describe_endpoint(EndpointName=endpoint_name)\nstatus = resp[\'EndpointStatus\']\nprint("Status: " + status)\n\nwhile status==\'Creating\':\n    time.sleep(60)\n    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)\n    status = resp[\'EndpointStatus\']\n    print("Status: " + status)\n\nprint("Arn: " + resp[\'EndpointArn\'])\nprint("Status: " + status)')

runtime_client = boto3.client('runtime.sagemaker')

import numpy as np
point_X = test_X[0]
point_X = np.expand_dims(point_X, axis=0)
point_y = test_y[0]
np.savetxt("test_point.csv", point_X, delimiter=",")

get_ipython().run_cell_magic('time', '', "import json\n\n\nfile_name = 'test_point.csv' #customize to your test file, will be 'mnist.single.test' if use data above\n\nwith open(file_name, 'r') as f:\n    payload = f.read().strip()\n\nresponse = runtime_client.invoke_endpoint(EndpointName=endpoint_name, \n                                   ContentType='text/csv', \n                                   Body=payload)\nresult = response['Body'].read().decode('ascii')\nprint('Predicted Class Probabilities: {}.'.format(result))")

floatArr = np.array(json.loads(result))
predictedLabel = np.argmax(floatArr)
print('Predicted Class Label: {}.'.format(predictedLabel))
print('Actual Class Label: {}.'.format(point_y))

sm_client.delete_endpoint(EndpointName=endpoint_name)

