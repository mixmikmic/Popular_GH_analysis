import numpy as np
# random seed for reproducibility
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
#Now we will import some utilities
from keras.utils import np_utils
#Fixed dimension ordering issue
from keras import backend as K
K.set_image_dim_ordering('th')

(X_train,y_train),(X_test, y_test)=cifar10.load_data()
#Preprocess imput data for Keras
# Reshape input data.
# reshape to be [samples][channels][width][height]
X_train=X_train.reshape(X_train.shape[0],3,32,32)
X_test=X_test.reshape(X_test.shape[0],3,32,32)
# to convert our data type to float32 and normalize our database
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
print(X_train.shape)
# Z-scoring or Gaussian Normalization
X_train=X_train - np.mean(X_train) / X_train.std()
X_test=X_test - np.mean(X_test) / X_test.std()

# convert 1-dim class arrays to 10 dim class metrices
#one hot encoding outputs
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]
print(num_classes)
#10
#Define a simple CNN model
print(X_train.shape)



model=Sequential()
model.add(Conv2D(32, (5,5), input_shape=(3,32,32), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))      # Dropout, one form of regularization
model.add(Flatten())
model.add(Dense(240,activation='elu'))
model.add(Dense(10, activation='softmax'))
print(model.output_shape)


model.compile(loss='binary_crossentropy', optimizer='adagrad')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200)
# Final evaluation of the model
scores =model.evaluate(X_test, y_test, verbose=0)
print('CNN error: % .2f%%' % (scores))



