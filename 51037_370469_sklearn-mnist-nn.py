get_ipython().magic('matplotlib inline')

from time import time
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:], cmap="gray")
    plt.title('Class: '+str(y_train[i]))

n_neighbors = 1
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X_train.reshape(-1,28*28), y_train)

t0 = time()
predictions = clf.predict(X_test[:100,:,:].reshape(-1,28*28))
print('Time elapsed: %.2fs' % (time()-t0))

print('Predicted', len(predictions), 'digits with accuracy:', accuracy_score(y_test[:100], predictions))

n_neighbors = 1
clf_reduced = neighbors.KNeighborsClassifier(n_neighbors)
clf_reduced.fit(X_train[:1024,:,:].reshape(-1,28*28), y_train[:1024])

t0 = time()
predictions_reduced = clf_reduced.predict(X_test.reshape(-1,28*28))
print('Time elapsed: %.2fs' % (time()-t0))

print('Predicted', len(predictions_reduced), 'digits with accuracy:', accuracy_score(y_test, predictions_reduced))

def show_failures(predictions, trueclass=None, predictedclass=None, maxtoshow=10):
    errors = predictions!=y_test
    print('Showing max', maxtoshow, 'first failures. '
          'The predicted class is shown first and the correct class in parenthesis.')
    ii = 0
    plt.figure(figsize=(maxtoshow, 1))
    for i in range(X_test.shape[0]):
        if ii>=maxtoshow:
            break
        if errors[i]:
            if trueclass is not None and y_test[i] != trueclass:
                continue
            if predictedclass is not None and predictions[i] != predictedclass:
                continue
            plt.subplot(1, maxtoshow, ii+1)
            plt.axis('off')
            plt.imshow(X_test[i,:,:], cmap="gray")
            plt.title("%d (%d)" % (predictions[i], y_test[i]))
            ii = ii + 1
            
show_failures(predictions_reduced)



