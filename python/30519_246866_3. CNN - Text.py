import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D

from sklearn.cross_validation import train_test_split

np.random.seed(2)

model_variation = 'CNN-rand'  #  CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)

# Model Hyperparameters
sequence_length = 56
embedding_dim = 20          
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

# Training parameters
batch_size = 32
num_epochs = 2

# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count                        
context = 10        # Context window size    

print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()


if model_variation=='CNN-non-static' or model_variation=='CNN-static':
    embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation=='CNN-static':
        x = embedding_weights[0][x]
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')    

data = np.append(x,y,axis = 1)

train, test = train_test_split(data, test_size = 0.15,random_state = 0)

X_test = test[:,:56]
Y_test = test[:,56:58]


X_train = train[:,:56]
Y_train = train[:,56:58]
train_rows = np.random.randint(0,X_train.shape[0],2500)
X_train = X_train[train_rows]
Y_train = Y_train[train_rows]

print("Vocabulary Size: {:d}".format(len(vocabulary)))

def initialize():
    
    global graph_in
    global convs
    
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []

#Buliding the first layer (Convolution Layer) of the network
def build_layer_1(filter_length):
    
   
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    return conv

#Adding a max pooling layer to the model(network)
def add_max_pooling(conv):
    
    pool = MaxPooling1D(pool_length=2)(conv)
    return pool

#Adding a flattening layer to the model(network), before adding a dense layer
def add_flatten(conv_or_pool):
    
    flatten = Flatten()(conv_or_pool)
    return flatten

def add_sequential(graph):
    
    #main sequential model
    model = Sequential()
    if not model_variation=='CNN-static':
        model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                        weights=embedding_weights))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    
    return model

#1.Convolution 2.Flatten
def one_layer_convolution():
    
    initialize()
    
    conv = build_layer_1(3)
    flatten = add_flatten(conv)
    
    convs.append(flatten)
    out = convs[0]

    graph = Model(input=graph_in, output=out)
    
    model = add_sequential(graph)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=1, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

#1.Convolution 2.Max Pooling 3.Flatten
def two_layer_convolution():
    
    initialize()
    
    conv = build_layer_1(3)
    pool = add_max_pooling(conv)
    flatten = add_flatten(pool)
    
    convs.append(flatten)
    out = convs[0]

    graph = Model(input=graph_in, output=out)
    
    model = add_sequential(graph)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=num_epochs, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

#1.Convolution 2.Max Pooling 3.Flatten 4.Convolution 5.Flatten
def three_layer_convolution():
    
    initialize()
    
    conv = build_layer_1(3)
    pool = add_max_pooling(conv)
    flatten = add_flatten(pool)
    
    convs.append(flatten)
    
    conv = build_layer_1(4)
    flatten = add_flatten(conv)
    
    convs.append(flatten)
    
    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)
    
    model = add_sequential(graph)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=1, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

#1.Convolution 2.Max Pooling 3.Flatten 4.Convolution 5.Max Pooling 6.Flatten
def four_layer_convolution():
    
    initialize()
    
    conv = build_layer_1(3)
    pool = add_max_pooling(conv)
    flatten = add_flatten(pool)
    
    convs.append(flatten)
    
    conv = build_layer_1(4)
    pool = add_max_pooling(conv)
    flatten = add_flatten(pool)
    
    convs.append(flatten)
    
    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)
    
    model = add_sequential(graph)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=num_epochs, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

get_ipython().run_cell_magic('time', '', '#1.Convolution 2.Flatten\none_layer_convolution()')

get_ipython().run_cell_magic('time', '', '#1.Convolution 2.Max Pooling 3.Flatten\ntwo_layer_convolution()')

get_ipython().run_cell_magic('time', '', '#1.Convolution 2.Max Pooling 3.Flatten 4.Convolution 5.Flatten\nthree_layer_convolution()')

get_ipython().run_cell_magic('time', '', '#1.Convolution 2.Max Pooling 3.Flatten 4.Convolution 5.Max Pooling 6.Flatten\nfour_layer_convolution()')

