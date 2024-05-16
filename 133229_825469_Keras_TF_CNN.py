import os
import sys
import numpy as np
os.environ['KERAS_BACKEND'] = "tensorflow"
import keras as K
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from common.params import *
from common.utils import *

# channels_last is faster
K.backend.set_image_data_format('channels_last')

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tensorflow.__version__)
print(K.backend.backend())
# Channels should be last (otherwise slow)
print(K.backend.image_data_format())
print("GPU: ", get_gpu_name())

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

def create_symbol():
    model = Sequential()
    
    model.add(Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))
    return model

def init_model(m):
    m.compile(
        loss = "categorical_crossentropy",
        optimizer = K.optimizers.SGD(LR, MOMENTUM),
        metrics = ['accuracy'])
    return m

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=False, one_hot=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')

get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')

model.summary()

get_ipython().run_cell_magic('time', '', '# Train model\nmodel.fit(x_train,\n          y_train,\n          batch_size=BATCHSIZE,\n          epochs=EPOCHS,\n          verbose=1)')

get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(x_test, batch_size=BATCHSIZE)\ny_guess = np.argmax(y_guess, axis=-1)\ny_truth = np.argmax(y_test, axis=-1)')

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

