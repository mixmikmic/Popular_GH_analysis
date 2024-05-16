get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

with open("trainFeatsv2.pckl", "rb") as f:
    trainFeats = pickle.load(f)
with open("trainLabelv2.pckl", "rb") as f:
    trainLabel = pickle.load(f)
    
with open("testFeatsv2.pckl", "rb") as f:
    testFeats = pickle.load(f)
with open("testLabelv2.pckl", "rb") as f:
    testLabel = pickle.load(f)

class mlp(nn.Module):
    def __init__(self,n_channels): #n_channels => length of feature vector
        super(mlp, self).__init__()
        self.L1 = nn.Linear(n_channels,6) #Mapping from input to hidden layer       
        self.L2 = nn.Linear(6,10) #Mapping from hidden layer to output
    def forward(self,x): #x => Input
        x = self.L1(x) #Feed-forward  
        x = F.relu(x) #Sigmoid non-linearity
        x = self.L2(x) #Feed-forward           
        x = F.softmax(x) #Sigmoid non-linearity
        return x

# Generating 1-hot label vectors
trainLabel2 = np.zeros((50000,10))
testLabel2 = np.zeros((10000,10))
for d1 in range(trainLabel.shape[0]):
    trainLabel2[d1,trainLabel[d1]] = 1
for d2 in range(testLabel.shape[0]):
    testLabel2[d2,testLabel[d2]] = 1

# Creating pytorch dataset from the feature matices
trainDataset = TensorDataset(torch.from_numpy(trainFeats), torch.from_numpy(trainLabel2))
testDataset = TensorDataset(torch.from_numpy(testFeats), torch.from_numpy(testLabel2))
# Creating dataloader
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)
testLoader = DataLoader(testDataset, batch_size=64, shuffle=True,num_workers=4, pin_memory=True)

# Checking availability of GPU
use_gpu = torch.cuda.is_available()

# Definining the training routine
def train_model(model,criterion,num_epochs,learning_rate):
        start = time.time()
        train_loss = [] #List for saving the loss per epoch     
        
        for epoch in range(num_epochs):
            epochStartTime = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            running_loss = 0.0           
            # Loading data in batches
            batch = 0
            for data in trainLoader:
                inputs,labels = data
                # Wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.float().cuda()),                         Variable(labels.float().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)    
                # Initializing model gradients to zero
                model.zero_grad() 
                # Data feed-forward through the network
                outputs = model(inputs)
                # Predicted class is the one with maximum probability
                _, preds = torch.max(outputs.data, 1)
                # Finding the MSE
                loss = criterion(outputs, labels)
                # Accumulating the loss for each batch
                running_loss += loss.data[0]
                # Backpropaging the error
                if batch == 0:
                    totalLoss = loss
                    totalPreds = preds
                    batch += 1                    
                else:
                    totalLoss += loss
                    totalPreds = torch.cat((totalPreds,preds),0)  
                    batch += 1
                    
            totalLoss = totalLoss/batch
            totalLoss.backward()
            
            # Updating the model parameters
            for f in model.parameters():
                f.data.sub_(f.grad.data * learning_rate)                
           
            epoch_loss = running_loss/50000  #Total loss for one epoch
            train_loss.append(epoch_loss) #Saving the loss over epochs for plotting the graph           
            
            print('Epoch loss: {:.6f}'.format(epoch_loss))
            epochTimeEnd = time.time()-epochStartTime
            print('Epoch complete in {:.0f}m {:.0f}s'.format(
            epochTimeEnd // 60, epochTimeEnd % 60))
            print('-' * 25)
            # Plotting Loss vs Epochs
            fig1 = plt.figure(1)        
            plt.plot(range(epoch+1),train_loss,'r--',label='train')        
            if epoch==0:
                plt.legend(loc='upper left')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
            fig1.savefig('mlp_lossPlot.png')             

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        return model

feat_length = 2+5+2
model = mlp(feat_length).cuda() # Initilaizing the model
criterion = nn.MSELoss() 
model = train_model(model,criterion,num_epochs=20,learning_rate=10) # Training the model

# Finding testing accuracy
test_running_corr = 0
# Loading data in batches
batches = 0
for tsData in testLoader:
    inputs,labels = tsData
    # Wrap them in Variable
    if use_gpu:
        inputs, labels = Variable(inputs.float().cuda()),             Variable(labels.float().cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)        
    # Feedforward train data batch through model
    output = model(inputs) 
    # Predicted class is the one with maximum probability
    _,preds = output.data.max(1)    
    if batches==0:
        totalPreds = preds
        batches = 1
    else:
        totalPreds = torch.cat((totalPreds,preds),0)

ts_corr = np.sum(np.equal(totalPreds.cpu().numpy(),testLabel))
ts_acc = ts_corr/10000.0
print('Testing accuracy = '+str(ts_acc))



