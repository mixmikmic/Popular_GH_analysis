import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding

# fix random seed
seed = 7
np.random.seed(seed)

from keras.datasets import imdb

(X_train, Y_train), (X_val, Y_val) = imdb.load_data(nb_words=5000)
print X_train.shape, X_train.dtype

print type(X_train[0])

from keras.preprocessing import sequence

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)

print len(X_train[0])

# define a simple model
def create_simple_model():
    model = Sequential()
    model.add(Embedding(5000, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

simple_model = create_simple_model()

simple_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=10, batch_size=30, verbose=1)

# define conv model
def create_conv_model():
    model = Sequential()
    model.add(Embedding(5000, 32, input_length=max_words))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

conv_model = create_conv_model()

conv_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=10, batch_size=30, verbose=1)

















