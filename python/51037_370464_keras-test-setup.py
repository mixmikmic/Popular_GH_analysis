get_ipython().magic('matplotlib inline')

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D 
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

model = Sequential()

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])

X_train = np.random.rand(128, 100)
Y_train = np_utils.to_categorical(np.random.randint(10, size=128))

model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=2);

X_test = np.random.rand(64, 100)
Y_test = np_utils.to_categorical(np.random.randint(10, size=64))

loss, acc = model.evaluate(X_test, Y_test, batch_size=32)
print()
print('loss:', loss, 'acc:', acc)



