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

#dataset = datasets.CIFAR10    # 170Mb of data download
dataset = datasets.CIFAR100   # 169Mb of data download

data_path = './data'

transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
           ])

train_loader = torch.utils.data.DataLoader(
    dataset(data_path, train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset(data_path, train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                print("Skipping the Batchnorm for these experiments")
                #layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            #else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def cifar10(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    return model

def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    return model

#model = cifar10(128)
#model = cifar10(32)
model = cifar100(32)
if use_cuda:
    model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#checkpoints_dir = './data/cache/overfitting/cifar10'
checkpoints_dir = './data/cache/overfitting/cifar100'

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
        #loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
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
        #tot_loss += F.nll_loss(output, target).data[0]
        tot_loss += F.cross_entropy(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    tot_loss = tot_loss  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tot_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return tot_loss / len(test_loader), correct / len(test_loader.dataset)    

epoch, losses_by_epoch = 0, []

for _ in range(100):
    epoch+=1
    train_loss, train_correct = train(epoch)
    save(epoch)
    test_loss, test_correct = test(epoch)
    losses_by_epoch.append( [ train_loss, train_correct, test_loss, test_correct ] )
print("Finished %d epochs" % (epoch,))

losses_by_epoch_np = np.array( losses_by_epoch )
np.save(os.path.join(checkpoints_dir, 'losses_by_epoch%03d.npy' % epoch), losses_by_epoch_np)

losses_by_epoch





