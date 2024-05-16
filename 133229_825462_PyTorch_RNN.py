import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.init as init
from torch import autograd
from torch.autograd import Variable
from common.params_lstm import *
from common.utils import *

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())

class SymbolModule(nn.Module):
    def __init__(self):
        super(SymbolModule, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=MAXFEATURES,
                                      embedding_dim=EMBEDSIZE)
        # If batch-first then input and output 
        # provided as (batch, seq, features)
        # Cudnn used by default if possible
        self.gru = nn.GRU(input_size=EMBEDSIZE, 
                          hidden_size=NUMHIDDEN, 
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False)   
        self.l_out = nn.Linear(in_features=NUMHIDDEN*1,
                               out_features=2)

    def forward(self, x):
        x = self.embedding(x)
        h0 = Variable(torch.zeros(1, BATCHSIZE, NUMHIDDEN)).cuda()
        x, h = self.gru(x, h0)  # outputs, states
        # just get the last output state
        x = x[:,-1,:].squeeze()
        x = self.l_out(x)
        return x

def init_model(m):
    opt = optim.Adam(m.parameters(), lr=LR, betas=(BETA_1, BETA_2), eps=EPS)
    criterion = nn.CrossEntropyLoss()
    return opt, criterion

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES)\n# Torch-specific\nx_train = x_train.astype(np.int64)\nx_test = x_test.astype(np.int64)\ny_train = y_train.astype(np.int64)\ny_test = y_test.astype(np.int64)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', 'sym = SymbolModule()\nsym.cuda() # CUDA!')

get_ipython().run_cell_magic('time', '', 'optimizer, criterion = init_model(sym)')

get_ipython().run_cell_magic('time', '', '# Sets training = True\nsym.train()  \nfor j in range(EPOCHS):\n    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Get samples\n        data = Variable(torch.LongTensor(data).cuda())\n        target = Variable(torch.LongTensor(target).cuda())\n        # Init\n        optimizer.zero_grad()\n        # Forwards\n        output = sym(data)\n        # Loss\n        loss = criterion(output, target)\n        # Back-prop\n        loss.backward()\n        optimizer.step()\n    # Log\n    print(j)')

get_ipython().run_cell_magic('time', '', '# Test model\n# Sets training = False\nsym.eval()\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, target in yield_mb(x_test, y_test, BATCHSIZE):\n    # Get samples\n    data = Variable(torch.LongTensor(data).cuda())\n    # Forwards\n    output = sym(data)\n    pred = output.data.max(1)[1].cpu().numpy().squeeze()\n    # Collect results\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred\n    c += 1')

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

