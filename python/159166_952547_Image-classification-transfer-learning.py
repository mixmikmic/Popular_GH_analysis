get_ipython().run_cell_magic('time', '', "import boto3\nimport re\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\n\nbucket='<<bucket-name>>' # customize to your bucket\n\ncontainers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest',\n              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest',\n              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest',\n              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest'}\ntraining_image = containers[boto3.Session().region_name]\nprint(training_image)")

import os
import urllib.request
import boto3

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

        
def upload_to_s3(channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)


# # caltech-256
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec')
download('http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec')
upload_to_s3('validation', 'caltech-256-60-val.rec')
upload_to_s3('train', 'caltech-256-60-train.rec')

# The algorithm supports multiple network depth (number of layers). They are 18, 34, 50, 101, 152 and 200
# For this training, we will use 18 layers
num_layers = 18
# we need to specify the input image shape for the training data
image_shape = "3,224,224"
# we also need to specify the number of training samples in the training set
# for caltech it is 15420
num_training_samples = 15420
# specify the number of output classes
num_classes = 257
# batch size for training
mini_batch_size =  128
# number of epochs
epochs = 2
# learning rate
learning_rate = 0.01
top_k=2
# Since we are using transfer learning, we set use_pretrained_model to 1 so that weights can be 
# initialized with pre-trained weights
use_pretrained_model = 1

get_ipython().run_cell_magic('time', '', 'import time\nimport boto3\nfrom time import gmtime, strftime\n\n\ns3 = boto3.client(\'s3\')\n# create unique job name \njob_name_prefix = \'sagemaker-imageclassification-notebook\'\ntimestamp = time.strftime(\'-%Y-%m-%d-%H-%M-%S\', time.gmtime())\njob_name = job_name_prefix + timestamp\ntraining_params = \\\n{\n    # specify the training docker image\n    "AlgorithmSpecification": {\n        "TrainingImage": training_image,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": \'s3://{}/{}/output\'.format(bucket, job_name_prefix)\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.p2.xlarge",\n        "VolumeSizeInGB": 50\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "image_shape": image_shape,\n        "num_layers": str(num_layers),\n        "num_training_samples": str(num_training_samples),\n        "num_classes": str(num_classes),\n        "mini_batch_size": str(mini_batch_size),\n        "epochs": str(epochs),\n        "learning_rate": str(learning_rate),\n        "use_pretrained_model": str(use_pretrained_model)\n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 360000\n    },\n#Training data should be inside a subdirectory called "train"\n#Validation data should be inside a subdirectory called "validation"\n#The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/train/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/validation/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None"\n        }\n    ]\n}\nprint(\'Training job name: {}\'.format(job_name))\nprint(\'\\nInput Data Location: {}\'.format(training_params[\'InputDataConfig\'][0][\'DataSource\'][\'S3DataSource\']))')

# create the Amazon SageMaker training job
sagemaker = boto3.client(service_name='sagemaker')
sagemaker.create_training_job(**training_params)

# confirm that the training job has started
status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print('Training job current status: {}'.format(status))

try:
    # wait for the job to finish and report the ending status
    sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
    training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = training_info['TrainingJobStatus']
    print("Training job ended with status: " + status)
except:
    print('Training failed to start')
     # if exception is raised, that means it has failed
    message = sagemaker.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))

training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
status = training_info['TrainingJobStatus']
print("Training job ended with status: " + status)

get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\nsage = boto3.Session().client(service_name=\'sagemaker\') \n\nmodel_name="test-image-classification-model"\nprint(model_name)\ninfo = sage.describe_training_job(TrainingJobName=job_name)\nmodel_data = info[\'ModelArtifacts\'][\'S3ModelArtifacts\']\nprint(model_data)\ncontainers = {\'us-west-2\': \'433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest\',\n              \'us-east-1\': \'811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest\',\n              \'us-east-2\': \'825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest\',\n              \'eu-west-1\': \'685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest\'}\nhosting_image = containers[boto3.Session().region_name]\nprimary_container = {\n    \'Image\': hosting_image,\n    \'ModelDataUrl\': model_data,\n}\n\ncreate_model_response = sage.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response[\'ModelArn\'])')

from time import gmtime, strftime

timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
endpoint_config_name = job_name_prefix + '-epc-' + timestamp
endpoint_config_response = sage.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.p2.xlarge',
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print('Endpoint configuration name: {}'.format(endpoint_config_name))
print('Endpoint configuration arn:  {}'.format(endpoint_config_response['EndpointConfigArn']))

get_ipython().run_cell_magic('time', '', "import time\n\ntimestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())\nendpoint_name = job_name_prefix + '-ep-' + timestamp\nprint('Endpoint name: {}'.format(endpoint_name))\n\nendpoint_params = {\n    'EndpointName': endpoint_name,\n    'EndpointConfigName': endpoint_config_name,\n}\nendpoint_response = sagemaker.create_endpoint(**endpoint_params)\nprint('EndpointArn = {}'.format(endpoint_response['EndpointArn']))")

# get the status of the endpoint
response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
status = response['EndpointStatus']
print('EndpointStatus = {}'.format(status))


# wait until the status has changed
sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)


# print the status of the endpoint
endpoint_response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
status = endpoint_response['EndpointStatus']
print('Endpoint creation ended with EndpointStatus = {}'.format(status))

if status != 'InService':
    raise Exception('Endpoint creation failed.')

import boto3
runtime = boto3.Session().client(service_name='runtime.sagemaker') 

get_ipython().system('wget -O /tmp/test.jpg http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/008.bathtub/008_0007.jpg')
file_name = '/tmp/test.jpg'
# test image
from IPython.display import Image
Image(file_name)  

import json
import numpy as np
with open(file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)
response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='application/x-image', 
                                   Body=payload)
result = response['Body'].read()
# result will be in json format and convert it to ndarray
result = json.loads(result)
# the result will output the probabilities for all classes
# find the class with maximum probability and print the class index
index = np.argmax(result)
object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))

sage.delete_endpoint(EndpointName=endpoint_name)



