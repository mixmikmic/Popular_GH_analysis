import os
import sys
import numpy as np
os.environ['KERAS_BACKEND'] = "tensorflow"
import keras as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, CuDNNGRU
from common.params_lstm import *
from common.utils import *

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Keras: ", K.__version__)
print("Numpy: ", np.__version__)
print("Tensorflow: ", tf.__version__)
print(K.backend.backend())

def create_symbol(CUDNN=True):
    model = Sequential()
    model.add(Embedding(MAXFEATURES, EMBEDSIZE, input_length=MAXLEN))
    # Only return last output
    if not CUDNN:
        model.add(GRU(NUMHIDDEN, return_sequences=False, return_state=False))
    else:
        model.add(CuDNNGRU(NUMHIDDEN, return_sequences=False, return_state=False))
    model.add(Dense(2, activation='softmax'))
    return model

def init_model(m):
    m.compile(
        loss = "categorical_crossentropy",
        optimizer = K.optimizers.Adam(LR, BETA_1, BETA_2, EPS),
        metrics = ['accuracy'])
    return m

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES, one_hot=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Load symbol\nsym = create_symbol()')

get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model(sym)')

model.summary()

get_ipython().run_cell_magic('time', '', '# Train model\nmodel.fit(x_train,\n          y_train,\n          batch_size=BATCHSIZE,\n          epochs=EPOCHS,\n          verbose=1)')

get_ipython().run_cell_magic('time', '', 'y_guess = model.predict(x_test, batch_size=BATCHSIZE)\ny_guess = np.argmax(y_guess, axis=-1)\ny_truth = np.argmax(y_test, axis=-1)')

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

