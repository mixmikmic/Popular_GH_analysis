get_ipython().magic('matplotlib inline')

import numpy as np
from sklearn import decomposition, feature_selection
from skimage.measure import block_reduce
from skimage.feature import canny

import matplotlib.pyplot as plt
import seaborn as sns

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)

get_ipython().run_cell_magic('time', '', "n_components = 50\npca = decomposition.PCA(n_components=n_components, whiten=True)\nX_pca = pca.fit_transform(X_train.reshape(-1,28*28))\nprint('X_pca:', X_pca.shape)")

plt.figure()
plt.plot(np.arange(n_components)+1, pca.explained_variance_)
plt.title('Explained variance by PCA components')

filter_size = 2
X_train_downsampled = block_reduce(X_train, 
                                   block_size=(1, filter_size, filter_size), 
                                   func=np.mean)
print('X_train:', X_train.shape)
print('X_train_downsampled:', X_train_downsampled.shape)

get_ipython().run_cell_magic('time', '', "sigma = 1.0\nX_train_canny = np.zeros(X_train.shape)\nfor i in range(X_train.shape[0]):\n    X_train_canny[i,:,:] = canny(X_train[i,:,:], sigma=sigma)\nprint('X_train_canny:', X_train_canny.shape)")

pltsize=1

plt.figure(figsize=(10*pltsize, pltsize))
plt.suptitle('Original')
plt.subplots_adjust(top=0.8)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:], cmap="gray", interpolation='none')

plt.figure(figsize=(10*pltsize, pltsize))
plt.suptitle('Downsampled with a %dx%d filter' % (filter_size, filter_size))
plt.subplots_adjust(top=0.8)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train_downsampled[i,:,:], cmap="gray", interpolation='none')
    
plt.figure(figsize=(10*pltsize, pltsize))
plt.suptitle('Canny edge detection with sigma=%.2f' % sigma)
plt.subplots_adjust(top=0.8)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train_canny[i,:,:], cmap="gray", interpolation='none')

variances = np.var(X_train.reshape(-1,28*28), axis=0)
plt.figure()
plt.plot(variances)
plt.title('Component-wise variance of MNIST digits')

plt.figure()
with sns.axes_style("white"):
    plt.imshow(variances.reshape(28,28), interpolation='none')
plt.title('Pixel-wise variance of MNIST digits')
plt.grid(False)

get_ipython().run_cell_magic('time', '', "variance_threshold = 1000\nlv = feature_selection.VarianceThreshold(threshold=variance_threshold)\nX_lv = lv.fit_transform(X_train.reshape(-1,28*28))\nprint('X_lv:', X_lv.shape)")

get_ipython().run_cell_magic('time', '', "k = 50\nukb = feature_selection.SelectKBest(k=k)\nX_ukb = ukb.fit_transform(X_train.reshape(-1,28*28), y_train)\nprint('X_ukb:', X_ukb.shape)")

support = ukb.get_support()
plt.figure()
with sns.axes_style("white"):
    plt.imshow(support.reshape(28,28), interpolation='none')
plt.title('Support of SelectKBest() with k=%d' % k)
plt.grid(False)

X_test_pca = pca.transform(X_test.reshape(-1,28*28))
print('X_test_pca:', X_test_pca.shape)



