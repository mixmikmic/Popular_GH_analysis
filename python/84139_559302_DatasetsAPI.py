get_ipython().magic('pylab inline')

import sklearn.datasets as datasets

# datasets.load_sample_images()
china = datasets.load_sample_image('china.jpg')

flower = datasets.load_sample_image('flower.jpg')

plt.imshow(china)

plt.imshow(flower)

flower.shape

get_ipython().magic('pinfo datasets.make_blobs')

X, y = datasets.make_blobs()

X.shape, y.shape

data = datasets.load_boston()

data.keys()

data.data.shape, data.target.shape

data.feature_names

print data.DESCR

faces = datasets.fetch_olivetti_faces()

faces.keys()

faces.images.shape, faces.data.shape, faces.target.shape



