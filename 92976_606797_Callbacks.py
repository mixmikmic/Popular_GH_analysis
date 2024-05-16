from keras.callbacks import ModelCheckpoint

mc = ModelCheckpoint(
    filepath='tmp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    period=5)

from keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor='val_loss',
    min_delta=0.01,
    patience=5,
    verbose=1,
    mode='max')

from keras.callbacks import LearningRateScheduler

lrs = LearningRateScheduler(lambda epoch: 1./epoch)

from keras.callbacks import ReduceLROnPlateau

rlrop = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1, 
    patience=10, 
    verbose=0, 
    mode='auto', 
    epsilon=0.0001, 
    cooldown=4, 
    min_lr=10e-7)

from keras.callbacks import CSVLogger

csvl = CSVLogger(
    filename='tmp/training.log',
    separator=',', 
    append=False)

from keras.callbacks import TensorBoard

TensorBoard(
    log_dir='./logs', 
    histogram_freq=0, 
    write_graph=True, 
    write_images=False,
    embeddings_freq=100,
    embeddings_layer_names=None, # this list of embedding layers...
    embeddings_metadata=None)      # with this metadata associated with them.)

from keras.callbacks import LambdaCallback

# Print the batch number at the beginning of every batch.
def print_batch(batch, logs):
    print batch
batch_print_callback = LambdaCallback(
    on_batch_begin=print_batch)

# Terminate some processes after having finished model training.
processes = []
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
    p.terminate() for p in processes if p.is_alive()])

import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

