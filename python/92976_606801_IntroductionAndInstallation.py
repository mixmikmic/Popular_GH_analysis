from keras.models import Sequential

model = Sequential()

# quickly grab the data
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_train.shape

from keras.layers import Dense

model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', 
              optimizer='adam',
              metrics=['mean_absolute_percentage_error'])

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=20, batch_size=404)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

loss_and_metrics

prices = model.predict(x_test, batch_size=128)

prices[:5]



