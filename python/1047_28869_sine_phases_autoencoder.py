get_ipython().magic('pylab inline')
import keras
import numpy as np
import keras

N = 50
# phase_step = 1 / (2 * np.pi)
t = np.arange(50)
phases = np.linspace(0, 1, N) * 2 * np.pi
x = np.array([np.sin(2 * np.pi / N * t + phi) for phi in phases])
print(x.shape)
imshow(x);

plot(x[0]);
plot(x[1]);
plot(x[2]);

from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder

encoder = containers.Sequential([
        Dense(25, input_dim=50),
        Dense(12)
    ])
decoder = containers.Sequential([
        Dense(25, input_dim=12),
        Dense(50)
    ])

model = Sequential()
model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

model.compile(loss='mean_squared_error', optimizer='sgd')

plot(model.predict(x)[0]);

from loss_history import LossHistory
loss_history = LossHistory()
model.fit(x, x, nb_epoch=1000, batch_size=50, callbacks=[loss_history])

plot(model.predict(x)[0])
plot(x[0])

plot(model.predict(x)[10])
plot(x[10])

print('last loss:', loss_history.losses[-1])
plot(loss_history.losses);

imshow(model.get_weights()[0], interpolation='nearest', cmap='gray');

imshow(model.get_weights()[2], interpolation='nearest', cmap='gray');

x_noised = x + 0.2 * np.random.random(len(x[0]))
plot(x_noised[0], label='input')
plot(model.predict(x_noised)[0], label='predicted')
legend();

x_shifted = np.cos(2*np.pi/N * t.reshape(1, -1))
plot(x_shifted[0], label='input')
plot(model.predict(x_shifted)[0], label='predicted')
legend();



