from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x = Input(shape=(784,))
# This works, and returns the 10-way softmax we defined above.
y = model(x)

from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)

from keras.layers import concatenate

x_in = Input(shape=(100,), name='x_in')
y_in = Input(shape=(100,), name='y_in')

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(x_in)
y = Dense(64, activation='relu')(y_in)

z = concatenate([x, y])

x = Dense(1, activation='sigmoid', name='x_out')(z)
y = Dense(10, activation='softmax', name='y_out')(z)

model = Model(inputs=[x_in, y_in], outputs=[x, y])

model.summary()

from keras.utils import to_categorical

import numpy as np
data = np.random.random((1000, 100))
xs = np.random.randint(2, size=(1000, 1))
ys = np.random.randint(10, size=(1000, 1))

model.compile(optimizer='rmsprop', loss=['binary_crossentropy', 'categorical_crossentropy'],
              loss_weights=[1., 0.2])

model.fit([data, data], [xs, to_categorical(ys)],
          epochs=1, batch_size=32)

model.compile(optimizer='rmsprop',
              loss={'x_out': 'binary_crossentropy', 'y_out': 'categorical_crossentropy'},
              loss_weights={'x_out': 1., 'y_out': 0.2})

# And trained it via:
model.fit({'x_in': data, 'y_in': data},
          {'x_out': xs, 'y_out': to_categorical(ys)},
          epochs=1, batch_size=32)

inputs = Input(shape=(64,))

# a layer instance is callable on a tensor, and returns a tensor
layer_we_share = Dense(64, activation='relu')

# Now we apply the layer twice
x = layer_we_share(inputs)
x = layer_we_share(x)

predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

a = Input(shape=(140, 256))

dense = Dense(32)
affine_a = dense(a)

assert dense.output == affine_a

a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

dense = Dense(32)
affine_a = dense(a)
affine_b = dense(b)

dense.output

assert dense.get_output_at(0) == affine_a
assert dense.get_output_at(1) == affine_b



