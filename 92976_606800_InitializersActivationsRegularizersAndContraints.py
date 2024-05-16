from keras.layers import Dense

get_ipython().magic('pinfo Dense')

layer = Dense(10, kernel_initializer='lecun_uniform', bias_initializer='ones')

from keras.initializers import Constant

layer = Dense(10, kernel_initializer='he_normal', bias_initializer=Constant(7))

from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

Dense(64, kernel_initializer=my_init)

from keras.layers import Activation, Dense, Input

x = Input((1,))
x = Dense(64)(x)
x = Activation('tanh')(x)

x = Input((1,))
x = Dense(64, activation='tanh')(x)

from keras import backend as K

x = Input((1,))
x = Dense(64, activation=K.tanh)(x)
x = Activation(K.tanh)(x)

from keras import regularizers
Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))

# available regularizers
regularizers.l1(0.)
regularizers.l2(0.)
regularizers.l1_l2(0.)

# Custom regularizer
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

Dense(64, input_dim=64,
                kernel_regularizer=l1_reg)

from keras.layers import ActivityRegularization

get_ipython().magic('pinfo ActivityRegularization')

from keras.constraints import max_norm

Dense(64, kernel_constraint=max_norm(2.))

from keras.engine.topology import Layer
from keras.activations import hard_sigmoid
from keras import regularizers
import numpy as np


class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      contraint='unit_norm',
                                      regularizer=regularizers.l1(1.),
                                      trainable=True)
        
        # Another way to enable this regularization is with the add loss function
        # self.add_loss(self.kernel, inputs=None)
        
        super(MyLayer, self).build(input_shape) 

    def call(self, x):
        return hard_sigmoid(K.dot(x, self.kernel))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



