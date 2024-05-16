import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

# fix random seed
seed = 7
np.random.seed(7)

# loaddata
dataframe = pd.read_csv("./data_set/sonar.data", header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]
encoder = LabelEncoder()
encoder.fit(Y)
Y_enc = encoder.transform(Y)

# define a model
def create_model():
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
    model.add(Dense(30, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

from keras.callbacks import LearningRateScheduler

# define a drop_rate function
def drop_rate(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

model = create_model()

# create a LearningRateScheduler instance
lrate = LearningRateScheduler(drop_rate)
model.fit(X, Y_enc, validation_split=0.2, nb_epoch=150, batch_size=10, callbacks=[lrate], verbose=2)



