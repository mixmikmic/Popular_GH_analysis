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

from sagemaker import KMeans

data_location = 's3://{}/kmeans_highlevel_example/data'.format(bucket)
output_location = 's3://{}/kmeans_example/output'.format(bucket)

print('training data will be uploaded to: {}'.format(data_location))
print('training artifacts will be uploaded to: {}'.format(output_location))

kmeans = KMeans(role=role,
                train_instance_count=2,
                train_instance_type='ml.c4.8xlarge',
                output_path=output_location,
                k=10,
                data_location=data_location)

get_ipython().run_cell_magic('time', '', '\nkmeans.fit(kmeans.record_set(train_set[0]))')

get_ipython().run_cell_magic('time', '', "\nkmeans_predictor = kmeans.deploy(initial_instance_count=1,\n                                 instance_type='ml.c4.xlarge')")

result = kmeans_predictor.predict(train_set[0][30:31])
print(result)

get_ipython().run_cell_magic('time', '', "\nresult = kmeans_predictor.predict(valid_set[0][0:100])\nclusters = [r.label['closest_cluster'].float32_tensor.values[0] for r in result]")

for cluster in range(10):
    print('\n\n\nCluster {}:'.format(int(cluster)))
    digits = [ img for l, img in zip(clusters, valid_set[0]) if int(l) == cluster ]
    height=((len(digits)-1)//5)+1
    width=5
    plt.rcParams["figure.figsize"] = (width,height)
    _, subplots = plt.subplots(height, width)
    subplots=numpy.ndarray.flatten(subplots)
    for subplot, image in zip(subplots, digits):
        show_digit(image, subplot=subplot)
    for subplot in subplots[len(digits):]:
        subplot.axis('off')

    plt.show()

print(kmeans_predictor.endpoint)

import sagemaker
sagemaker.Session().delete_endpoint(kmeans_predictor.endpoint)

