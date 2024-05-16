get_ipython().magic('matplotlib inline')

from time import time

import numpy as np
from minisom import MiniSom

from pylab import text,show,cm,axis,figure,subplot,imshow,zeros
import matplotlib.pyplot as plt
import seaborn as sns

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Let's inspect only 1024 first training samples in this notebook
X = X_train[:1024]
y = y_train[:1024]

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X:', X.shape)
print('y:', y.shape)

xsize = 16
ysize = 10
epochs = 20

t0 = time()
som = MiniSom(xsize, ysize, 28*28 ,sigma=.5, learning_rate=0.2)
som.train_random(X.reshape(-1,28*28), X.shape[0]*epochs)
print('Time elapsed: %.2fs' % (time()-t0))

t0 = time()
wmap = {}
qerrors = np.empty((xsize,ysize))
qerrors.fill(np.nan)
for im,x in enumerate(X.reshape(-1,28*28)):
    (i,j) = som.winner(x)
    qe = np.linalg.norm(x-som.weights[i,j])
    if np.isnan(qerrors[i,j]) or qe<qerrors[i,j]:
        wmap[(i,j)] = im
        qerrors[i,j] = qe
print('Time elapsed: %.2fs' % (time()-t0))

figure(1)
for j in range(ysize): # images mosaic
	for i in range(xsize):
		if (i,j) in wmap:
			text(i+.5, j+.5, str(y[wmap[(i,j)]]), 
                 color=cm.Dark2(y[wmap[(i,j)]]/9.), 
                 fontdict={'weight': 'bold', 'size': 11})
ax = axis([0,som.weights.shape[0],0,som.weights.shape[1]])

figure(facecolor='white')
cnt = 0
for j in reversed(range(ysize)):
	for i in range(xsize):
		subplot(ysize,xsize,cnt+1,frameon=False, xticks=[], yticks=[])
		if (i,j) in wmap:
			imshow(X[wmap[(i,j)]])
		else:
			imshow(zeros((28,28)))
		cnt = cnt + 1

figure(facecolor='white')
cnt = 0
for j in reversed(range(ysize)):
	for i in range(xsize):
		subplot(ysize,xsize,cnt+1,frameon=False, xticks=[], yticks=[])
		imshow(som.weights[i,j].reshape(28,28))
		cnt = cnt + 1



