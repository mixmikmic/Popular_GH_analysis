get_ipython().magic('pylab inline')
import keras
import numpy as np

t = np.arange(50).reshape(1, -1)
x = np.sin(2*np.pi/50*t)
print(x.shape)
plot(t[0], x[0]);

from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder

encoder = containers.Sequential([Dense(25, input_dim=50), Dense(12)])
decoder = containers.Sequential([Dense(25, input_dim=12), Dense(50)])

model = Sequential()
model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

model.compile(loss='mean_squared_error', optimizer='sgd')

# prediction with initial weight should be random
plot(model.predict(x)[0]);

# train the model and store the loss values as function of time
from loss_history import LossHistory
loss_history = LossHistory()
model.fit(x, x, nb_epoch=500, batch_size=1, callbacks=[loss_history])

plot(loss_history.losses);

plot(log10(loss_history.losses));

plot(model.predict(x)[0])
plot(x[0]);

x_noised = x + 0.2 * np.random.random(len(x[0]))
plot(x_noised[0], label='input')
plot(model.predict(x_noised)[0], label='predicted')
legend();

x_shifted = np.cos(2*np.pi/50*t)
plot(x_shifted[0], label='input')
plot(model.predict(x_shifted)[0], label='predicted')
legend();

x_scaled = 0.2 * x
plot(x_scaled[0], label='input')
plot(model.predict(x_scaled)[0], label='predicted')
legend();

