bucket = '<your_s3_bucket_name_here>'
prefix = 'sagemaker/pca-mnist'
 
# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

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

import io
import numpy as np
import sagemaker.amazon.common as smac

vectors = np.array([t.tolist() for t in train_set[0]]).T

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors)
buf.seek(0)

get_ipython().run_cell_magic('time', '', "import boto3\nimport os\n\nkey = 'recordio-pb-data'\nboto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)\ns3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)\nprint('uploaded training data location: {}'.format(s3_train_data))")

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))

containers = {'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:latest',
              'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/pca:latest',
              'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/pca:latest',
              'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/pca:latest'}

import boto3
import sagemaker

sess = sagemaker.Session()

pca = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.c4.xlarge',
                                    output_path=output_location,
                                    sagemaker_session=sess)
pca.set_hyperparameters(feature_dim=50000,
                        num_components=10,
                        subtract_mean=True,
                        algorithm_mode='randomized',
                        mini_batch_size=200)

pca.fit({'train': s3_train_data})

pca_predictor = pca.deploy(initial_instance_count=1,
                           instance_type='ml.c4.xlarge')

from sagemaker.predictor import csv_serializer, json_deserializer

pca_predictor.content_type = 'text/csv'
pca_predictor.serializer = csv_serializer
pca_predictor.deserializer = json_deserializer

result = pca_predictor.predict(train_set[0][:, 0])
print(result)

import numpy as np

eigendigits = []
for array in np.array_split(train_set[0].T, 50):
    result = pca_predictor.predict(array)
    eigendigits += [r['projection'] for r in result['projections']]


eigendigits = np.array(eigendigits).T

for e in enumerate(eigendigits):
    show_digit(e[1], 'eigendigit #{}'.format(e[0]))

import sagemaker

sagemaker.Session().delete_endpoint(pca_predictor.endpoint)

