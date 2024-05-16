import os
import sys
import caffe2
import numpy as np
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew, optimizer, utils
from caffe2.proto import caffe2_pb2
from common.params import *
from common.utils import *

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())

if GPU:
    device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)  # Run on GPU
else:
    device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)  # Run on CPU

def create_model(m, device_opts) :
    with core.DeviceScope(device_opts):
        conv1 = brew.conv(m, 'data', 'conv1', dim_in=3, dim_out=50, kernel=3, pad=1, no_gradient_to_input=1)
        relu1 = brew.relu(m, conv1, 'relu1')
        conv2 = brew.conv(m, relu1, 'conv2', dim_in=50, dim_out=50, kernel=3, pad=1)
        pool1 = brew.max_pool(m, conv2, 'pool1', kernel=2, stride=2)
        relu2 = brew.relu(m, pool1, 'relu2')
        drop1 = brew.dropout(m, relu2, 'drop1', ratio=0.25)

        conv3 = brew.conv(m, drop1, 'conv3', dim_in=50, dim_out=100, kernel=3, pad=1)
        relu3 = brew.relu(m, conv3, 'relu3')
        conv4 = brew.conv(m, relu3, 'conv4', dim_in=100, dim_out=100, kernel=3, pad=1)
        pool2 = brew.max_pool(m, conv4, 'pool2', kernel=2, stride=2)   
        relu4 = brew.relu(m, pool2, 'relu4')
        drop2 = brew.dropout(m, relu4, 'drop2', ratio=0.25)
        
        fc1 = brew.fc(m, drop2, 'fc1', dim_in=100 * 8 * 8, dim_out=512)
        relu5 = brew.relu(m, fc1, 'relu5')
        drop3 = brew.dropout(m, relu5, 'drop3', ratio=0.5)
        
        fc2 = brew.fc(m, drop3, 'fc2', dim_in=512, dim_out=N_CLASSES)
        softmax = brew.softmax(m, fc2, 'softmax')
        return softmax

def add_training_operators(softmax, m, device_opts) :
    with core.DeviceScope(device_opts):
        xent = m.LabelCrossEntropy([softmax, "label"], 'xent')
        loss = m.AveragedLoss(xent, "loss")
        #brew.accuracy(m, [softmax, "label"], "accuracy")
        m.AddGradientOperators([loss])
        opt = optimizer.build_sgd(
            m,
            base_learning_rate=LR, 
            policy='fixed',
            momentum=MOMENTUM)

def init_model():
    # Create Place-holder for data
    workspace.FeedBlob("data", x_train[:BATCHSIZE], device_option=device_opts)
    workspace.FeedBlob("label", y_train[:BATCHSIZE], device_option=device_opts)
    
    # Initialise model
    train_arg_scope = {
        'order': 'NCHW',
        'use_cudnn': True,
        'cudnn_exhaustive_search': True,
        'ws_nbytes_limit': (64 * 1024 * 1024),
    }
    train_model = model_helper.ModelHelper(
        name="train_net", arg_scope=train_arg_scope
    )
    softmax = create_model(train_model, device_opts=device_opts)
    add_training_operators(softmax, train_model, device_opts=device_opts)

    # Initialise workspace
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)
    return train_model

get_ipython().run_cell_magic('time', '', '# Data into format for library\nx_train, x_test, y_train, y_test = cifar_for_library(channel_first=True)\nprint(x_train.shape, x_test.shape, y_train.shape, y_test.shape)\nprint(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)')

get_ipython().run_cell_magic('time', '', '# Initialise model\nmodel = init_model()')

get_ipython().run_cell_magic('time', '', '# Train model\nfor j in range(EPOCHS):\n    for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):\n        # Run one mini-batch at time\n        workspace.FeedBlob("data", data, device_option=device_opts)\n        workspace.FeedBlob("label", label, device_option=device_opts)\n        workspace.RunNet(model.net)       \n    print("Finished epoch: ", j)\n    print(str(j) + \': \' + str(workspace.FetchBlob("loss")))')

get_ipython().run_cell_magic('time', '', '# Init test model\ntest_arg_scope = {\n    \'order\': \'NCHW\',\n    \'use_cudnn\': True,\n    \'cudnn_exhaustive_search\': True,\n    \'ws_nbytes_limit\': (64 * 1024 * 1024),\n    \'is_test\': True,\n}\ntest_model= model_helper.ModelHelper(name="test_net", init_params=False, arg_scope=test_arg_scope)\ncreate_model(test_model, device_opts=device_opts)\nworkspace.RunNetOnce(test_model.param_init_net)\nworkspace.CreateNet(test_model.net, overwrite=True)\n\n# Run test\nn_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE\ny_guess = np.zeros(n_samples, dtype=np.int)\ny_truth = y_test[:n_samples]\nc = 0\nfor data, label in yield_mb(x_test, y_test, BATCHSIZE):\n    workspace.FeedBlob("data", data, device_option=device_opts)\n    workspace.RunNet(test_model.net)\n    y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = (np.argmax(workspace.FetchBlob("softmax"), axis=-1))\n    c += 1')

print("Accuracy: ", sum(y_guess == y_truth)/float(len(y_guess)))

