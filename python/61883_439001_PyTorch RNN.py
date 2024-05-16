get_ipython().magic('matplotlib inline')
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn

X = np.arange(-10,10,0.1)
X.shape

## Code taken from http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

rolling_window(X[:5], 2)

X = np.arange(-10,10,0.1)
X = np.cos(np.mean(rolling_window(X, 5), -1))
#X = X[:-5+1]
print(X.shape)

plt.plot(X)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.h2o = nn.Linear(input_size+hidden_size, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x, h):
        inp = torch.cat((x,h), 1)
        hidden = self.tanh(self.i2h(inp))
        output = self.h2o(inp)
        return hidden, output
    
    
    def get_output(self, X):
        time_steps = X.size(0)
        batch_size = X.size(1)
        hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if torch.cuda.is_available() and X.is_cuda:
            hidden = hidden.cuda()
        outputs = []
        hiddens = []
        for t in range(time_steps):
            hidden, output = self.forward(X[t], hidden)
            outputs.append(output)
            hiddens.append(hidden)
        return torch.cat(hiddens, 1), torch.cat(outputs, 1)
    
## Helper functions

def get_variable_from_np(X):
    return Variable(torch.from_numpy(X)).float()


def get_training_data(X, batch_size=10, look_ahead=1):
    ## Lookahead will always be one as the prediction is for 1 step ahead
    inputs = []
    targets = []
    time_steps = X.shape[0]
    for i in range(0, time_steps-batch_size-look_ahead):
        inp = X[i:i+batch_size, np.newaxis, np.newaxis]
        inputs.append(get_variable_from_np(inp))
        target = X[i+look_ahead:i+batch_size+look_ahead, np.newaxis, np.newaxis]
        targets.append(get_variable_from_np(target))
        #print(inp.shape, target.shape)
    return torch.cat(inputs, 1), torch.cat(targets, 1)

print(torch.cat([get_variable_from_np(X[i:i+5, np.newaxis, np.newaxis]) for i in range(X.shape[0]-5-1)], 1).size())
print(torch.cat([get_variable_from_np(X[i:i+5, np.newaxis, np.newaxis]) for i in range(1, X.shape[0]-5)], 1).size())

inputs, targets = get_training_data(X, batch_size=5)
inputs.size(), targets.size()

inputs, targets = get_training_data(X, batch_size=3)
inputs.size(), targets.size()

rnn = RNN(30, 20, 1)

criterion = nn.MSELoss()
batch_size = 10
TIMESTEPS = 5

batch = Variable(torch.randn(batch_size, 30))
hidden = Variable(torch.randn(batch_size, 20))
target = Variable(torch.randn(batch_size, 1))

loss = 0
for t in range(TIMESTEPS):
    hidden, output = rnn(batch, hidden)
    loss += criterion(output, target)
    
loss.backward()    

rnn = RNN(30, 20, 1).cuda()

criterion = nn.MSELoss()
batch_size = 10
TIMESTEPS = 5

batch = Variable(torch.randn(batch_size, 30)).cuda()
hidden = Variable(torch.randn(batch_size, 20)).cuda()
target = Variable(torch.randn(batch_size, 1)).cuda()

loss = 0
for t in range(TIMESTEPS):
    hidden, output = rnn(batch, hidden)
    loss += criterion(output, target)
    
loss.backward()

rnn = RNN(1,3,1).cuda()

criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(rnn.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adadelta(rnn.parameters())

X[:, np.newaxis, np.newaxis].shape

batch = get_variable_from_np(X[:, np.newaxis, np.newaxis]).cuda()

batch.is_cuda

batch = get_variable_from_np(X[:, np.newaxis, np.newaxis]).cuda()
hiddens, outputs = rnn.get_output(batch)

outputs.size()

target = get_variable_from_np(X[np.newaxis, :])
target.size()

torch.cat([get_variable_from_np(X[i:i+10, np.newaxis, np.newaxis]) for i in range(5)], 1).size()

torch.cat([get_variable_from_np(X[i:i+10, np.newaxis, np.newaxis]) for i in range(5)], 1).size()

rnn = RNN(1,3,1)
if torch.cuda.is_available():
    rnn = rnn.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adadelta(rnn.parameters())

batch_size = 1
TIMESTEPS = X.shape[0]
epochs = 10000
print_every = 1000
inputs, targets = get_training_data(X, batch_size=100)
if torch.cuda.is_available() and rnn.is_cuda:
    inputs = inputs.cuda()
    targets = targets.cuda()
print(inputs.size(), targets.size())
losses = []

for i in range(epochs):
    optimizer.zero_grad()
    hiddens, outputs = rnn.get_output(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.data[0])
    if (i+1) % print_every == 0:
        print("Loss at epoch [%s]: %.3f" % (i, loss.data[0]))

inputs, targets = get_training_data(X, batch_size=5)

inputs.size()

inputs = inputs.cuda()

torch.cuda.is_available()

outputs[:, 0].size()

X.shape, outputs[:, 0].data.numpy().flatten().shape

plt.plot(X, '-b', label='data')
plt.plot(outputs[:, 0].data.numpy().flatten(), '-r', label='rnn') # add some offset to view each curve
plt.legend()

input_size, hidden_size, output_size = 1,3,1
lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
output_layer = nn.Linear(hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD([
    {"params": lstm.parameters()},
    {"params": output_layer.parameters()}
], lr=0.001, momentum=0.9)

lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
output_layer = nn.Linear(hidden_size, output_size)

batch = get_variable_from_np(X[:, np.newaxis, np.newaxis])
batch.size()

hidden = Variable(torch.zeros(1, batch.size(1), hidden_size))
cell_state = Variable(torch.zeros(1, batch.size(1), hidden_size))

hx = (hidden, cell_state)
output, (h_n, c_n) = lstm.forward(batch, hx)

output.size()

out = output_layer.forward(output[0])
out

criterion = nn.MSELoss()
optimizer = torch.optim.SGD([
    {"params": lstm.parameters()},
    {"params": output_layer.parameters()}
], lr=0.001, momentum=0.9)

batch_size = 1
epochs = 10

inputs, targets = get_training_data(X, max_steps=1)

for i in range(epochs):
    loss = 0
    optimizer.zero_grad()
    hidden = Variable(torch.zeros(1, inputs.size(1), hidden_size))
    cell_state = Variable(torch.zeros(1, inputs.size(1), hidden_size))
    hx = (hidden, cell_state)
    output, (h_n, c_n) = lstm.forward(inputs, hx)
    losses = []
    for j in range(output.size()[0]):
        out = output_layer.forward(output[j])
        losses.append((out - targets[j])**2)
        #loss += criterion(out, target[i])
    loss = torch.mean(torch.cat(losses, 1))
    loss.backward()
    optimizer.step()
    print("Loss at epoch [%s]: %.3f" % (i, loss.squeeze().data[0]))

output.size()

out.size()

targets.size()

y_pred = []
for i in range(output.size()[1]):
        out = output_layer.forward(output[i])
        y_pred.append(out.squeeze().data[0])
y_pred = np.array(y_pred)

plt.plot(X, '-b', alpha=0.5, label='data')
plt.plot(y_pred + 0.1, '-r', label='rnn')
plt.legend()

y_pred



