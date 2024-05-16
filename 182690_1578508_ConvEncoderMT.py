from __future__ import unicode_literals, print_function, division
from io import open
from collections import namedtuple
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

use_cuda = torch.cuda.is_available() # To check if GPU is available
MAX_LENGTH = 10 # We restrict our experiments to sentences of length 10 or less
embedding_size = 256
hidden_size_gru = 256
attn_units = 256
conv_units = 256
num_iterations = 750
print_every = 100
batch_size = 1
sample_size = 1000
dropout = 0.2
encoder_layers = 3
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# Function to convert unicdoe string to plain ASCII
# Thanks to http://stackoverflow.com/a/518232/2809427

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Takes all unicode characters, converts them to ascii
# Replaces full stop with space full stop (so that Fire!
# becomes Fire !)
# Removes everything apart from alphabet characters and
# stop characters.

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# Returns the cuda tensor type of a variable if CUDA is available
def check_and_convert_to_cuda(var):
    return var.cuda() if use_cuda else var

data = pd.read_csv('data/eng-fra.txt', sep='\t', names=['english', 'french'])
data = data[data.english.str.lower().str.startswith(eng_prefixes)].iloc[:sample_size]

data['english'] = data.apply(lambda row: normalizeString(row.english), axis=1)
data['french'] = data.apply(lambda row: normalizeString(row.french), axis=1)

Vocabulary = namedtuple('Vocabulary', ['word2id', 'id2word']) # A Named tuple representing the vocabulary of a particular language

def construct_vocab(sentences):
    word2id = dict()
    id2word = dict()
    word2id[SOS_TOKEN] = 0
    word2id[EOS_TOKEN] = 1
    id2word[0] = SOS_TOKEN
    id2word[1] = EOS_TOKEN
    for sentence in sentences:
        for word in sentence.strip().split(' '):
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word[len(word2id)-1] = word
    return Vocabulary(word2id, id2word)

english_vocab = construct_vocab(data.english)
french_vocab = construct_vocab(data.french)

def sent_to_word_id(sentences, vocab, eos=True):
    data = []
    for sent in sentences:
        if eos:
            end = [vocab.word2id[EOS_TOKEN]]
        else:
            end = []
        words = sent.strip().split(' ')
        
        if len(words) < MAX_LENGTH:
            data.append([vocab.word2id[w] for w in words] + end)
    return data

english_data = sent_to_word_id(data.english, english_vocab)
french_data = sent_to_word_id(data.french, french_vocab)

input_dataset = [Variable(torch.LongTensor(sent)) for sent in french_data]
output_dataset = [Variable(torch.LongTensor(sent)) for sent in english_data]

if use_cuda: # And if cuda is available use the cuda tensor types
    input_dataset = [i.cuda() for i in input_dataset]
    output_dataset = [i.cuda() for i in output_dataset]

get_ipython().run_cell_magic('script', 'false', 'class ConvEncoder(nn.Module):\n    def __init__(self, vocab_size, embedding_size, dropout=0.2,\n                 num_channels_attn=512, num_channels_conv=512, max_len=MAX_LENGTH,\n                 kernel_size=3, num_layers=5):\n      pass\n    def forward(self, position_ids, sentence_as_wordids):\n      # position_ids refer to position of individual words in the sentence \n      # represented by sentence_as_wordids. \n      pass')

get_ipython().run_cell_magic('script', 'false', 'class ConvEncoder(nn.Module):\n    def __init__(self, vocab_size, embedding_size, dropout=0.2,\n                 num_channels_attn=512, num_channels_conv=512, max_len=MAX_LENGTH,\n                 kernel_size=3, num_layers=5):\n      super(ConvEncoder, self).__init__()\n      # Here we define the required layers that would be used in the forward pass\n      self.position_embedding = nn.Embedding(max_len, embedding_size)\n      self.word_embedding = nn.Embedding(vocab_size, embedding_size)\n      self.num_layers = num_layers\n      self.dropout = dropout\n      \n      # Convolution Layers\n      self.conv = nn.ModuleList([nn.Conv1d(num_channels_conv, num_channels_conv, kernel_size,\n                                      padding=kernel_size // 2) for _ in range(num_layers)])\n      \n    def forward(self, position_ids, sentence_as_wordids):\n      # position_ids refer to position of individual words in the sentence \n      # represented by sentence_as_wordids. \n      pass')

class ConvEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.2,
                 num_channels_attn=512, num_channels_conv=512, max_len=MAX_LENGTH,
                 kernel_size=3, num_layers=5):
        super(ConvEncoder, self).__init__()
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv = nn.ModuleList([nn.Conv1d(num_channels_conv, num_channels_conv, kernel_size,
                                      padding=kernel_size // 2) for _ in range(num_layers)])

    def forward(self, position_ids, sentence_as_wordids):
        # Retrieving position and word embeddings 
        position_embedding = self.position_embedding(position_ids)
        word_embedding = self.word_embedding(sentence_as_wordids)
        
        # Applying dropout to the sum of position + word embeddings
        embedded = F.dropout(position_embedding + word_embedding, self.dropout, self.training)
        
        # Transform the input to be compatible for Conv1d as follows
        # Length * Channel ==> Num Batches * Channel * Length
        embedded = torch.unsqueeze(embedded.transpose(0, 1), 0)
        
        # Successive application of convolution layers followed by residual connection
        # and non-linearity
        
        cnn = embedded
        for i, layer in enumerate(self.conv):
          # layer(cnn) is the convolution operation on the input cnn after which
          # we add the original input creating a residual connection
          cnn = F.tanh(layer(cnn)+cnn)        

        return cnn

get_ipython().run_cell_magic('script', 'false', '\nclass AttnDecoder(nn.Module):\n  def __init__(self, output_vocab_size, hidden_size_gru, embedding_size,\n               n_layers_gru):\n    \n    # This will generate the embedding g_i of previous output y_i\n    self.embedding = nn.Embedding(output_size, embedding_size)\n    \n    # A GRU \n    self.gru = nn.GRU(hidden_size_gru+embedding_size, hidden_size, n_layers_gru)\n    \n    # Dense layer for output transformation\n    self.dense_o = nn.Linear(hidden_size_gru, output_vocab_size)\n    \n  def forward(self, y_i, h_i, cnn_a, cnn_c):\n    \n    # generates the embedding of previous output\n    g_i = self.embedding(y_i)\n    \n    gru_output, gru_hidden = self.gru(torch.concat(g_i, input_context), h_i)\n    # gru_output: contains the output at each time step from the last layer of gru\n    # gru_hidden: contains hidden state of every layer of gru at the end\n    \n    # We want to compute a softmax over the last output of the last layer\n    output = F.log_softmax(self.dense_o(gru_hidden[-1]))\n    \n    # We return the softmax-ed output. We also need to collect the hidden state of the GRU\n    # to be used as h_i in the next forward pass\n    \n    return output, gru_hidden')

class AttnDecoder(nn.Module):
  
  def __init__(self, output_vocab_size, dropout = 0.2, hidden_size_gru = 128,
               cnn_size = 128, attn_size = 128, n_layers_gru=1,
               embedding_size = 128, max_sentece_len = MAX_LENGTH):

    super(AttnDecoder, self).__init__()
    
    self.n_gru_layers = n_layers_gru
    self.hidden_size_gru = hidden_size_gru
    self.output_vocab_size = output_vocab_size
    self.dropout = dropout
    
    self.embedding = nn.Embedding(output_vocab_size, hidden_size_gru)
    self.gru = nn.GRU(hidden_size_gru + embedding_size, hidden_size_gru,
                      n_layers_gru)
    self.transform_gru_hidden = nn.Linear(hidden_size_gru, embedding_size)
    self.dense_o = nn.Linear(hidden_size_gru, output_vocab_size)

    self.n_layers_gru = n_layers_gru
    
  def forward(self, y_i, h_i, cnn_a, cnn_c):
    
    g_i = self.embedding(y_i)
    g_i = F.dropout(g_i, self.dropout, self.training)
    
    d_i = self.transform_gru_hidden(h_i) + g_i
    a_i = F.softmax(torch.bmm(d_i, cnn_a).view(1, -1))
  
    c_i = torch.bmm(a_i.view(1, 1, -1), cnn_c.transpose(1, 2))
    gru_output, gru_hidden = self.gru(torch.cat((g_i, c_i), dim=-1), h_i)
    
    gru_hidden = F.dropout(gru_hidden, self.dropout, self.training)
    softmax_output = F.log_softmax(self.dense_o(gru_hidden[-1]))
    
    return softmax_output, gru_hidden


  # function to initialize the hidden layer of GRU. 
  def initHidden(self):
    result = Variable(torch.zeros(self.n_layers_gru, 1, self.hidden_size_gru))
    if use_cuda:
        return result.cuda()
    else:
        return result

def init_weights(m):
  
    if not hasattr(m, 'weight'):
        return
    if type(m) == nn.Conv1d:
        width = m.weight.data.shape[-1]/(m.weight.data.shape[0]**0.5)
    else:
        width = 0.05
        
    m.weight.data.uniform_(-width, width)


encoder_a = ConvEncoder(len(french_vocab.word2id), embedding_size, dropout=dropout,
                        num_channels_attn=attn_units, num_channels_conv=conv_units,
                        num_layers=encoder_layers)
encoder_c = ConvEncoder(len(french_vocab.word2id), embedding_size, dropout=dropout,
                        num_channels_attn=attn_units, num_channels_conv=conv_units,
                        num_layers=encoder_layers)
decoder = AttnDecoder(len(english_vocab.word2id), dropout = dropout,
                       hidden_size_gru = hidden_size_gru, embedding_size = embedding_size,
                       attn_size = attn_units, cnn_size = conv_units)

if use_cuda:
    encoder_a = encoder_a.cuda()
    encoder_c = encoder_c.cuda()
    decoder = decoder.cuda()

encoder_a.apply(init_weights)
encoder_c.apply(init_weights)
decoder.apply(init_weights)

encoder_a.training = True
encoder_c.training = True
decoder.training = True

def trainIters(encoder_a, encoder_c, decoder, n_iters, batch_size=32, learning_rate=1e-4, print_every=100):
  
    encoder_a_optimizer = optim.Adam(encoder_a.parameters(), lr=learning_rate)
    encoder_c_optimizer = optim.Adam(encoder_c.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    # Sample a training pair
    training_pairs = list(zip(*(input_dataset, output_dataset)))
    
    criterion = nn.NLLLoss()
    
    
    print_loss_total = 0
    
    # The important part of the code is the 3rd line, which performs one training
    # step on the batch. We are using a variable `print_loss_total` to monitor
    # the loss value as the training progresses
    
    for itr in range(1, n_iters + 1):
        training_pair = random.sample(training_pairs, k=batch_size)
        input_variable, target_variable = list(zip(*training_pair))
        
        loss = train(input_variable, target_variable, encoder_a, encoder_c,
                     decoder, encoder_a_optimizer, encoder_c_optimizer, decoder_optimizer,
                     criterion, batch_size=batch_size)
        
        print_loss_total += loss

        if itr % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print(print_loss_avg)
            print_loss_total=0
    print("Training Completed")

def train(input_variables, output_variables, encoder_a, encoder_c, decoder,
          encoder_a_optimizer, encoder_c_optimizer, decoder_optimizer, criterion, 
          max_length=MAX_LENGTH, batch_size=32):
    
  # Initialize the gradients to zero
  encoder_a_optimizer.zero_grad()
  encoder_c_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  for count in range(batch_size):
    # Length of input and output sentences
    input_variable = input_variables[count]
    output_variable = output_variables[count]

    input_length = input_variable.size()[0]
    output_length = output_variable.size()[0]

    loss = 0

    # Encoder outputs: We use this variable to collect the outputs
    # from encoder after each time step. This will be sent to the decoder.
    position_ids = Variable(torch.LongTensor(range(0, input_length)))
    position_ids = position_ids.cuda() if use_cuda else position_ids
    cnn_a = encoder_a(position_ids, input_variable)
    cnn_c = encoder_c(position_ids, input_variable)
    
    cnn_a = cnn_a.cuda() if use_cuda else cnn_a
    cnn_c = cnn_c.cuda() if use_cuda else cnn_c

    prev_word = Variable(torch.LongTensor([[0]])) #SOS
    prev_word = prev_word.cuda() if use_cuda else prev_word

    decoder_hidden = decoder.initHidden()

    for i in range(output_length):
      decoder_output, decoder_hidden =           decoder(prev_word, decoder_hidden, cnn_a, cnn_c)
      topv, topi = decoder_output.data.topk(1)
      ni = topi[0][0]
      prev_word = Variable(torch.LongTensor([[ni]]))
      prev_word = prev_word.cuda() if use_cuda else prev_word
      loss += criterion(decoder_output,output_variable[i])

      if ni==1: #EOS
        break

  # Backpropagation
  loss.backward()
  encoder_a_optimizer.step()
  decoder_optimizer.step()

  return loss.data[0]/output_length

trainIters(encoder_a,encoder_c, decoder, num_iterations, print_every=print_every, batch_size=batch_size)

def evaluate(sent_pair, encoder_a, encoder_c, decoder, source_vocab, target_vocab, max_length=MAX_LENGTH):
    source_sent = sent_to_word_id(np.array([sent_pair[0]]), source_vocab)
    if(len(source_sent) == 0):
        return
    source_sent = source_sent[0]
    input_variable = Variable(torch.LongTensor(source_sent))
    
    if use_cuda:
        input_variable = input_variable.cuda()
        
    input_length = input_variable.size()[0]
    position_ids = Variable(torch.LongTensor(range(0, input_length)))
    position_ids = position_ids.cuda() if use_cuda else position_ids
    cnn_a = encoder_a(position_ids, input_variable)
    cnn_c = encoder_c(position_ids, input_variable)
    cnn_a = cnn_a.cuda() if use_cuda else cnn_a
    cnn_c = cnn_c.cuda() if use_cuda else cnn_c
    
    prev_word = Variable(torch.LongTensor([[0]])) #SOS
    prev_word = prev_word.cuda() if use_cuda else prev_word

    decoder_hidden = decoder.initHidden()
    target_sent = []
    ni = 0
    out_length = 0
    while not ni==1 and out_length < 10:
        decoder_output, decoder_hidden =             decoder(prev_word, decoder_hidden, cnn_a, cnn_c)

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        target_sent.append(target_vocab.id2word[ni])
        prev_word = Variable(torch.LongTensor([[ni]]))
        prev_word = prev_word.cuda() if use_cuda else prev_word
        out_length += 1
        
    print("Source: " + sent_pair[0])
    print("Translated: "+' '.join(target_sent))
    print("Expected: "+sent_pair[1])
    print("")

encoder_a.training = False
encoder_c.training = False
decoder.training = False
samples = data.sample(n=100)
for (i, row) in samples.iterrows():
    evaluate((row.french, row.english), encoder_a, encoder_c, decoder, french_vocab, english_vocab)



