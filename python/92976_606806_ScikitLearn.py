from keras.models import Sequential
from keras.layers import Dense, Activation

def build_model(optimizer='rmsprop', dense_dims=32):
    model = Sequential()
    model.add(Dense(dense_dims, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model
    

from keras.wrappers.scikit_learn import KerasClassifier

keras_classifier = KerasClassifier(build_model, epochs=2)

import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

keras_classifier.fit(data, labels)

keras_classifier.predict_proba(data[:2])

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(keras_classifier, {'epochs': [2, 3], 'dense_dims':[16, 32]})

gs.fit(data, labels)

gs.best_params_



