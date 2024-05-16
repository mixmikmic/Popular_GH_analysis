# Generate dummy data
import numpy as np
data = np.random.random((2, 5))
labels = np.random.randint(3, size=(2, 3))

data, labels

from keras.utils import to_categorical, normalize

to_categorical(labels, num_classes=4)

normalize(data, order=1)

sequences = [
    [1,2,4,4],
    [3],
    [5,6,4,2,1,7,7,4,3],
    [3,3,4,3,2]
]

from keras.preprocessing.sequence import pad_sequences

padded_sequences = pad_sequences(sequences, maxlen=None, dtype='int32',
    padding='pre', truncating='post', value=0.)

padded_sequences

from keras.preprocessing.sequence import skipgrams

grams = skipgrams(padded_sequences[0], vocabulary_size=8,
    window_size=1, negative_samples=1., shuffle=True,
    categorical=False)

grams 

text ="""
    My name is Nathaniel.
    I like data science.
    Let's do deep learning.
    Keras, my fave lib.
    """

from keras.preprocessing.text import text_to_word_sequence

words = text_to_word_sequence(text, lower=True, split=" ")

words

# we can change the filter chars too
text_to_word_sequence(text, filters="'", lower=True, split=" ")

from keras.preprocessing.text import one_hot

one_hot(text, n=8, lower=True, split=" ")

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=None, lower=True, split=" ")

# tokenizer.fit_on_sequences
tokenizer.fit_on_texts([text])

tokenizer.texts_to_sequences([text])

tokenizer.texts_to_matrix([text], 'count')

tokenizer.texts_to_matrix(['Data Science is fun'], 'count')

from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


dataset = load_sample_images()

plt.imshow(dataset.images[0])

plt.imshow(dataset.images[1])

from keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None,
    preprocessing_function=None)

idg.fit(dataset.images)

import numpy
it = idg.flow(numpy.array(dataset.images), numpy.array([1, 1,]), batch_size=1)

plt.imshow(numpy.array(next(it)[0][0, :, :, :], dtype='uint8'))

# finally there is the option to flow_from_directory



