import os
os.environ['CHAINER_TYPE_CHECK'] = '0'

import sys
import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
from common.params import *
from common.utils import *

cuda.set_max_workspace_size(512 * 1024 * 1024)

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Chainer: ", chainer.__version__)
print("CuPy: ", chainer.cuda.cupy.__version__)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())

class SymbolModule(chainer.Chain):
    def __init__(self):
        super(SymbolModule, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 50, ksize=3, pad=1)
            self.conv2 = L.Convolution2D(50, 50, ksize=3, pad=1)
            self.conv3 = L.Convolution2D(50, 100, ksize=3, pad=1)
            self.conv4 = L.Convolution2D(100, 100, ksize=3, pad=1)
            # feature map size is 8*8 by pooling
            self.fc1 = L.Linear(100*8*8, 512)
            self.fc2 = L.Linear(512, N_CLASSES)
    
    def __call__(self, x):
        h = F.relu(self.conv2(F.relu(self.conv1(x))))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, 0.25)
        
        h = F.relu(self.conv4(F.relu(self.conv3(h))))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, 0.25)       
        
        h = F.dropout(F.relu(self.fc1(h)), 0.5)
        return self.fc2(h)

def init_model(m):
    optimizer = optimizers.MomentumSGD(lr=LR, momentum=MOMENTUM)
    optimizer.setup(m)
    return optimizer

get_ipython().run_cell_magic('time', '', '# Data into format for library\n#x_train, x_test, y_train, y_test = mnist_for_library(channel_first=True)\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Create symbol\nsym = SymbolModule()\nif GPU:\n    chainer.cuda.get_device(0).use()  # Make a specified GPU current\n    sym.to_gpu()  # Copy the model to the GPU')

get_ipython().run_cell_magic('time', '', 'optimizer = init_model(sym)')

get_ipython().run_cell_magic('time', '', 'for j in range(EPOCHS):\n    for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Get samples\n        data = cuda.to_gpu(data)\n        target = cuda.to_gpu(target)\n        output = sym(data)\n        loss = F.softmax_cross_entropy(output, target)\n        sym.cleargrads()\n        loss.backward()\n        optimizer.update()\n    # Log\n    print(j)')

get_ipython().run_cell_magic('time', '', "n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\n\nwith chainer.using_config('train', False), chainer.using_config('enable_backprop', False):\n    for data, target in yield_mb(x_test, y_test, BATCHSIZE):\n        # Forwards\n        pred = cuda.to_cpu(sym(cuda.to_gpu(data)).data.argmax(-1))\n        # Collect results\n        y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred\n        c += 1")

print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))

