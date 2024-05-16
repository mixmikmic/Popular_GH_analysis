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
from torch.autograd import Variable
from common.params import *
from common.utils import *

# Big impact on training-time (from 350 to 165s)
torch.backends.cudnn.benchmark=True # enables cudnn's auto-tuner

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())

class SymbolModule(nn.Module):
    def __init__(self):
        super(SymbolModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(50, 100, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        # feature map size is 8*8 by pooling
        self.fc1 = nn.Linear(100*8*8, 512)
        self.fc2 = nn.Linear(512, N_CLASSES)

    def forward(self, x):
        """ PyTorch requires a flag for training in dropout """
        x = self.conv2(F.relu(self.conv1(x)))
        x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        x = F.dropout(x, 0.25, training=self.training)

        x = self.conv4(F.relu(self.conv3(x)))
        x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        x = F.dropout(x, 0.25, training=self.training)

        x = x.view(-1, 100*8*8)   # reshape Variable
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=self.training)
        # nn.CrossEntropyLoss() contains softmax, don't apply twice
        #return F.log_softmax(x)
        return self.fc2(x)

def init_model(m):
    # Implementation of momentum:
    # v = \rho * v + g \\
    # p = p - lr * v
    opt = optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()
    return opt, criterion

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\n# Torch-specific\ny_train = y_train.astype(np.int64)\ny_test = y_test.astype(np.int64)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', 'sym = SymbolModule()\nsym.cuda() # CUDA!')

get_ipython().run_cell_magic('time', '', 'optimizer, criterion = init_model(sym)')

get_ipython().run_cell_magic('time', '', '# Sets training = True\nsym.train()  \nfor j in range(EPOCHS):\n    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Get samples\n        data = Variable(torch.FloatTensor(data).cuda())\n        target = Variable(torch.LongTensor(target).cuda())\n        # Init\n        optimizer.zero_grad()\n        # Forwards\n        output = sym(data)\n        # Loss\n        loss = criterion(output, target)\n        # Back-prop\n        loss.backward()\n        optimizer.step()\n    # Log\n    print(j)')

get_ipython().run_cell_magic('time', '', '# Test model\n# Sets training = False\nsym.eval()\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, target in yield_mb(x_test, y_test, BATCHSIZE):\n    # Get samples\n    data = Variable(torch.FloatTensor(data).cuda())\n    # Forwards\n    output = sym(data)\n    pred = output.data.max(1)[1].cpu().numpy().squeeze()\n    # Collect results\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred\n    c += 1')

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

