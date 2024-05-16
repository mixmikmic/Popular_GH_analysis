import numpy as np
import keras

#Import necessary nlp tools from gensim
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec

from random import shuffle

# Define source files for input data
source_dict = {'test-neg.txt':'TEST_NEG',
                'test-pos.txt':'TEST_POS',
                'train-neg.txt':'TRAIN_NEG',
                'train-pos.txt':'TRAIN_POS'
               }



# Define a LabeledDocSentence class to process multiple documents. This is an extension of the gensim's 
# LabeledLine class. Gensim's LabeledLine class does not process multiple documents, hence we need to define our own
# implementation.
class LabeledDocSentence():
    
    # Initialize the source dict
    def __init__(self, source_dict):
        self.sources = source_dict
    
    # This creates sentences as a list of words and assigns each sentence a tag 
    # e.g. [['word1', 'word2', 'word3', 'lastword'], ['label1']]
    def create_sentences(self):
        self.sentences = []
        for source_file, prefix in self.sources.items():
            with utils.smart_open(source_file) as f:
                for line_id, line in enumerate(f):
                    sentence_label = prefix + '_' + str(line_id)
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [sentence_label]))
        
        return self.sentences
             
    # Return a permutation of the sentences in each epoch. I read that this leads to the best results and 
    # helps the model to train properly.
    def get_permuted_sentences(self):
        sentences = self.create_sentences()
        shuffled = list(sentences)
        shuffle(shuffled)
        return shuffled

labeled_doc = LabeledDocSentence(source_dict) 
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

# Let the model learn the vocabulary - all the words in the paragraph
model.build_vocab(labeled_doc.get_permuted_sentences())

# Train the model on the entire set of sentences/reviews for 10 epochs. At each epoch sample a different
# permutation of the sentences to make the model learn better.
for epoch in range(10):
    print epoch
    model.train(labeled_doc.get_permuted_sentences(), total_examples=model.corpus_count, epochs=10)

# To avoid retraining, we save the model
model.save('imdb.d2v')

# Load the saved model
model_saved = Doc2Vec.load('imdb.d2v')

# Check what the model learned. It will show 10 most similar words to the input word. Since we kept the window size
# to be 10, it will show the 10 most recent.
model_saved.most_similar('good')

# Our model is a Doc2Vec model, hence it also learnt the sentence vectors apart from the word embeddings. Hence we
# can see the vector of any sentence by passing the tag for the sentence.
model_saved.docvecs['TRAIN_NEG_0']

# Create a labelled training and testing set

x_train = np.zeros((25000, 100))
y_train = np.zeros(25000)
x_test = np.zeros((25000, 100))
y_test = np.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    x_train[i] = model_saved.docvecs[prefix_train_pos]
    x_train[12500 + i] = model_saved.docvecs[prefix_train_neg]
    
    y_train[i] = 1
    y_train[12500 + i] = 0
    
    
for i in range(12500):
    prefix_test_pos = 'TRAIN_POS_' + str(i)
    prefix_test_neg = 'TRAIN_NEG_' + str(i)
    x_test[i] = model_saved.docvecs[prefix_test_pos]
    x_test[12500 + i] = model_saved.docvecs[prefix_test_neg]
    
    y_test[i] = 1
    y_test[12500 + i] = 0

print x_train

# Convert the output to a categorical variable to be used for the 2 neuron output layer in the neural network.

from keras.utils import to_categorical

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Create a neural network with a single hidden layer and a softmax output layer with 2 neurons.

from keras.models import Sequential
from keras.layers import Dense

nnet = Sequential()
nnet.add(Dense(32, input_dim=100, activation='relu'))
nnet.add(Dense(2, input_dim=32, activation='softmax'))
nnet.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Visualize the neural net's layer
nnet.summary()

# Train the net on the training data
nnet.fit(x_train, y_train_cat, epochs=5, batch_size=32)

# Predict on the test set
score = nnet.evaluate(x_test, y_test_cat, batch_size=32)
score[1]

