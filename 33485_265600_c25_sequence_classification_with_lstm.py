import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed
seed = 7
np.random.seed(seed)

# load dataset, only keep the top 5000 words, zero the rest
top_words = 5000
(X_train, Y_train), (X_val, Y_val) = imdb.load_data(nb_words=top_words)

# pad input sequence
maxlen = 500
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_val = sequence.pad_sequences(X_val, maxlen=maxlen)

Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_val = Y_val.reshape(Y_train.shape[0], 1)

# define and build a model
def create_model():
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=maxlen, dropout=0.2))
    model.add(LSTM(64, stateful=False, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

lstm = create_model()

# train model
lstm.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=64, nb_epoch=3, verbose=1)

# evaluate mode
scores = lstm.evaluate(X_val, Y_val)
print("Acc: %.2f%%"%(scores[1]*100))





