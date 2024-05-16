from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(100,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid', name='output')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

history = model.fit(data, labels, epochs=2)

model.summary()

model.get_config()

model.get_layer('output')

model.layers

history.history

model.history.history

from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)

json_string

from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)

config = model.get_config()
model = Model.from_config(config)

for weight in model.get_weights():
    print weight.shape

model.set_weights(model.get_weights())

get_ipython().magic('pinfo model.save_weights')

get_ipython().magic('pinfo model.load_weights')

from keras.models import load_model

get_ipython().magic('pinfo model.save')

get_ipython().magic('pinfo load_model')



