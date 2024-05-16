import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

use_cuda=torch.cuda.is_available()

random_seed = 42
batch_size = 200

#learning_rate, momentum = 0.01, 0.5  # SGD with momentum
learning_rate = 0.001   # SGD+Adam

log_interval = 20 # Num of batches between log messages

import numpy as np

import os
import time

torch.manual_seed(random_seed)
if use_cuda:
    torch.cuda.manual_seed(random_seed)

mnist_data_path = './data'

mnist_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(mnist_data_path, train=True, download=True, transform=mnist_transform),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(mnist_data_path, train=False, download=True, transform=mnist_transform),
    batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if use_cuda:
    model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

checkpoints_dir = './data/cache/overfitting/mnist'

#torch.save(the_model.state_dict(), PATH)

#the_model = TheModelClass(*args, **kwargs)
#the_model.load_state_dict(torch.load(PATH))

def save(epoch):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'saved_%03d.model' % (epoch+1, )))

def train(epoch):
    model.train()
    t0 = time.time()
    tot_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if True:
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            tot_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            bi1 = batch_idx+1
            print('Train Epoch: {} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.4f}\tt_epoch: {:.2f}secs'.format(
                epoch, bi1 * len(data), len(train_loader.dataset),
                100. * bi1 / len(train_loader), loss.data[0], 
                (time.time()-t0)*len(train_loader)/bi1,))
            
    tot_loss = tot_loss # loss function already averages over batch size
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        tot_loss / len(train_loader), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return tot_loss / len(train_loader), correct / len(train_loader.dataset)

def test(epoch):
    model.eval()
    tot_loss, correct = 0, 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        tot_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    tot_loss = tot_loss  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tot_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return tot_loss / len(test_loader), correct / len(test_loader.dataset)    

losses_by_epoch = []

for epoch in range(100):
    train_loss, train_correct = train(epoch+1)
    save(epoch+1)
    test_loss, test_correct = test(epoch+1)
    losses_by_epoch.append( [ train_loss, train_correct, test_loss, test_correct ] )
print("Finished %d epochs" % (epoch+1,))

losses_by_epoch_np = np.array( losses_by_epoch )
np.save(os.path.join(checkpoints_dir, 'losses_by_epoch.npy'), losses_by_epoch_np)

losses_by_epoch



