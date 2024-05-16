from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# set random seed
seed = 7
np.random.seed(seed)

# load data
dataset = np.loadtxt("./data_set/pima-indians-diabetes.data", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# set random seed
seed = 7
np.random.seed(seed)

# load data
dataset = np.loadtxt("./data_set/pima-indians-diabetes.data", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

# split data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=150, batch_size=10)

# evaluate model
scores = model.evaluate(X_val, Y_val, verbose=0)
print model.metrics_names
print 'val loss:', scores[0], 'val acc:', scores[0]

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np

seed = 7
np.random.seed(seed)

dataset = np.loadtxt("./data_set/pima-indians-diabetes.data", delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for train_idx, val_idx in kfold.split(X, Y):
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, nb_epoch=150, batch_size=10, verbose=0)
    scores = model.evaluate(X_val, Y_val, verbose=0)
    
    print("%s: %.2f%%"%(model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)

print("%.2f%% (+/- %.2f%%)"%(np.mean(cvscores), np.std(cvscores)))





























































