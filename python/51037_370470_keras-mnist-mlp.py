get_ipython().magic('matplotlib inline')

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
nb_classes = 10

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one-hot encoding:
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('Y_train:', Y_train.shape)

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(X_train[i,:,:], cmap="gray")
    plt.title('Class: '+str(y_train[i]))
    print('Training sample',i,': class:',y_train[i], ', one-hot encoded:', Y_train[i])

linmodel = Sequential()
linmodel.add(Dense(units=10, input_dim=28*28, activation='softmax'))

linmodel.compile(loss='categorical_crossentropy', 
                 optimizer='sgd', 
                 metrics=['accuracy'])
print(linmodel.summary())

SVG(model_to_dot(linmodel, show_shapes=True).create(prog='dot', format='svg'))

get_ipython().run_cell_magic('time', '', 'epochs = 10 # one epoch takes about 3 seconds\n\nlinhistory = linmodel.fit(X_train.reshape((-1,28*28)), \n                          Y_train, \n                          epochs=epochs, \n                          batch_size=32,\n                          verbose=2)')

plt.figure(figsize=(8,5))
plt.plot(linhistory.epoch,linhistory.history['loss'])
plt.title('loss')

plt.figure(figsize=(8,5))
plt.plot(linhistory.epoch,linhistory.history['acc'])
plt.title('accuracy');

linscores = linmodel.evaluate(X_test.reshape((-1,28*28)), 
                              Y_test, 
                              verbose=2)
print("%s: %.2f%%" % (linmodel.metrics_names[1], linscores[1]*100))

def show_failures(predictions, trueclass=None, predictedclass=None, maxtoshow=10):
    rounded = np.argmax(predictions, axis=1)
    errors = rounded!=y_test
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
            plt.title("%d (%d)" % (rounded[i], y_test[i]))
            ii = ii + 1

linpredictions = linmodel.predict(X_test.reshape((-1,28*28)))

show_failures(linpredictions)

x = np.arange(-4,4,.01)
plt.figure()
plt.plot(x, np.maximum(x,0), label='relu')
plt.plot(x, 1/(1+np.exp(-x)), label='sigmoid')
plt.plot(x, np.tanh(x), label='tanh')
plt.axis([-4, 4, -1.1, 1.5])
plt.title('Activation functions')
l = plt.legend()

# Model initialization:
model = Sequential()

# A simple model:
model.add(Dense(units=20, input_dim=28*28))
model.add(Activation('relu'))

# A bit more complex model:
#model.add(Dense(units=50, input_dim=28*28))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))

#model.add(Dense(units=50))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))

# The last layer needs to be like this:
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
print(model.summary())

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

get_ipython().run_cell_magic('time', '', 'epochs = 10 # one epoch with simple model takes about 4 seconds\n\nhistory = model.fit(X_train.reshape((-1,28*28)), \n                    Y_train, \n                    epochs=epochs, \n                    batch_size=32,\n                    verbose=2)')

plt.figure(figsize=(8,5))
plt.plot(history.epoch,history.history['loss'])
plt.title('loss')

plt.figure(figsize=(8,5))
plt.plot(history.epoch,history.history['acc'])
plt.title('accuracy');

get_ipython().run_cell_magic('time', '', 'scores = model.evaluate(X_test.reshape((-1,28*28)), Y_test, verbose=2)\nprint("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))')

predictions = model.predict(X_test.reshape((-1,28*28)))

show_failures(predictions)

show_failures(predictions, trueclass=6)



