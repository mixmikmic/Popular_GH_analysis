get_ipython().magic('matplotlib inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Using PyTorch version:', torch.__version__, 'CUDA:', torch.cuda.is_available())



