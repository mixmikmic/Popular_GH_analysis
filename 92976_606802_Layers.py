from keras.layers import Dense, Input

inputs = Input((8,))
layer = Dense(8)

layer.get_weights()

x = layer(inputs)

layer.name

layer.__class__.__name__

layer.trainable = True

layer.get_weights()

import numpy as np

new_bias = np.array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.])
layer.set_weights([layer.get_weights()[0], new_bias])

layer.get_weights()

layer.input, layer.output, layer.input_shape, layer.output_shape

x = layer(x)

layer.input, layer.output, layer.input_shape, layer.output_shape

layer.get_input_at(0), layer.get_output_at(0), layer.get_input_shape_at(0), layer.get_output_shape_at(0)

layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
config

from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})

from keras.layers import Lambda
from keras import backend as K

Lambda(lambda x: x ** 2)

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

Lambda(antirectifier, output_shape=antirectifier_output_shape)

import tensorflow as tf
Lambda(tf.reduce_mean, output_shape=(1,))

def to_pow(x, po=2):
    return x ** po

Lambda(to_pow, arguments={'po':5})

from keras.engine.topology import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)







