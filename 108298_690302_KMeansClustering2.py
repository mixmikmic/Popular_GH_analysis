get_ipython().magic('matplotlib inline')

from sklearn import datasets, cluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# note: I deliberately chose a random seed that ends up 
# labeling the clusters with the same numbering convention 
# as the original y values 
np.random.seed(2)

# load data
iris = datasets.load_iris()

X_iris = iris.data
y_iris = iris.target

# do the clustering
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris) 
labels = k_means.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y_iris == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y_iris.size))

# plot the clusters in color
fig = plt.figure(1, figsize=(8, 8))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=8, azim=200)
plt.cla()

ax.scatter(X_iris[:, 3], X_iris[:, 0], X_iris[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')

plt.show()



