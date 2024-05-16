import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, NonlinearityLayer, DenseLayer

from load_data import load_data

# For reproducibility
np.random.seed(23455)

# To enable the GPU, run the following code
import theano.gpuarray
theano.config.floatX = 'float32'
theano.gpuarray.use('cuda')

# Define symbolic inputs
x = T.matrix('x')
y = T.ivector('y')

nonlinearity = lasagne.nonlinearities.tanh

## Build the architecture of the network
# Input
input_var = x.reshape((-1, 1, 28, 28))
layer0 = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

# First conv / pool / nonlinearity block
conv1 = Conv2DLayer(layer0, num_filters=20, filter_size=(5, 5), nonlinearity=None)
pool1 = MaxPool2DLayer(conv1, pool_size=(2, 2))
act1 = NonlinearityLayer(pool1, nonlinearity=nonlinearity)

# Second conv / pool / nonlinearity block
conv2 = Conv2DLayer(act1, num_filters=50, filter_size=(5, 5), nonlinearity=None)
pool2 = MaxPool2DLayer(conv2, pool_size=(2, 2))
act2 = NonlinearityLayer(pool2, nonlinearity=nonlinearity)

# Fully-connected layer
dense1 = DenseLayer(act2, num_units=500, nonlinearity=nonlinearity)

# Fully-connected layer for the output
network = DenseLayer(dense1, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

## Training
# Prediction and cost
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, y)
loss = loss.mean()

# Gradients and updates
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=0.1)
train_fn = theano.function([x, y], loss, updates=updates)

## Monitoring and evaluation
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, y)
test_loss = test_loss.mean()

# Misclassification rate
test_err = T.mean(T.neq(T.argmax(test_prediction, axis=1), y),
                  dtype=theano.config.floatX)

valid_fn = theano.function([x, y], test_err)

def evaluate_model(train_fn, valid_fn, datasets, n_epochs, batch_size):
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size
    n_test_batches = test_set_x.shape[0] // batch_size

    ## early-stopping parameters
    # look as this many examples regardless
    patience = 10000
    # wait this much longer when a new best is found
    patience_increase = 2
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    # Go through this many minibatches before checking the network
    # on the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = %i' % iter)
            cost_ij = train_fn(train_set_x[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                               train_set_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size])

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [valid_fn(valid_set_x[i * batch_size:(i + 1) * batch_size],
                                              valid_set_y[i * batch_size:(i + 1) * batch_size])
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *                         improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        valid_fn(test_set_x[i * batch_size:(i + 1) * batch_size],
                                 test_set_y[i * batch_size:(i + 1) * batch_size])
                        for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))  

datasets = load_data('mnist.pkl.gz')

evaluate_model(train_fn, valid_fn, datasets, n_epochs=200, batch_size=500)



