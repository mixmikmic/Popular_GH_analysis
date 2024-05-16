import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

from keras.datasets import cifar10

# load dataset
(X_train, Y_train), (X_val, Y_val) = cifar10.load_data()

print X_train.shape, X_train.dtype
print Y_train.shape, Y_train.dtype

for i in range(0, 9):
    plt.subplot(331 + i)
    img = np.rollaxis(X_train[i], 0, 3)  # change axis ordering to [height][width][chanel]
    plt.imshow(img)
plt.show()

# normalize inputs from 0-255 to 0-1
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# one hot vector
Y_train = np_utils.to_categorical(Y_train.reshape(Y_train.shape[0],))
Y_val = np_utils.to_categorical(Y_val.reshape(Y_val.shape[0], ))
num_classes = Y_val.shape[1]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import backend
backend.set_image_dim_ordering('th')

# fix random seed
seed = 7
np.random.seed(seed)

# definea a CNN model
def create_cnn():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same',
                            activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# build the model
cnn = create_cnn()

# fit model
cnn.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=1, batch_size=32, verbose=1)

# evaluate model
scores = cnn.evaluate(X_val, Y_val, verbose=0)
print('Val_Acc: %.2f%%'%(scores[1]*100))







