import mxnet as mx

import numpy as np
import cmath
import graphviz
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--tau', type=float, default=1e-30)
parser.add_argument('--randomout', type=str, default="True")
parser.add_argument('--network', type=str, default="inception-28-small")
parser.add_argument('-f', type=str, default='')
args = parser.parse_args()
args.f = ''

# setup logging
import logging
logging.getLogger().handlers = []
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#logging.root = logging.getLogger(str(args))
logging.root = logging.getLogger()
logging.debug("test")

import importlib
softmax = importlib.import_module('symbol_' + args.network).get_symbol(10)

# If you'd like to see the network structure, run the plot_network function
a = mx.viz.plot_network(symbol=softmax.get_internals(),node_attrs={'shape':'rect','fixedsize':'false'},
                       shape={"data":(1,3, 28, 28)}) 

a.body.extend(['rankdir=RL', 'size="40,5"'])
#a

mx.random.seed(args.seed)
num_epoch = args.epochs
batch_size = args.batch_size
num_devs = 1
model = mx.model.FeedForward(ctx=[mx.gpu(i) for i in range(num_devs)], symbol=softmax, num_epoch = num_epoch,
                             learning_rate=0.1, momentum=0.9, wd=0.00001
                             ,optimizer=mx.optimizer.Adam()
                            )

import get_data
get_data.GetCifar10()

train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,28,28),
        batch_size=batch_size,
        preprocess_threads=4)
# test iterator make batch of 128 image, and center crop each image into 3x28x28 from original 3x32x32
# Note: We don't need round batch in test because we only test once at one time
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,28,28),
        batch_size=batch_size,
        round_batch=False,
        preprocess_threads=4)



from mxnet.ndarray import NDArray
from mxnet.base import NDArrayHandle
from mxnet import ndarray

class RandomOutMonitor(mx.monitor.Monitor):
    
    def __init__(self, initializer, network, tau=0.000001, *args,**kwargs):
        mx.monitor.Monitor.__init__(self, 1, *args, **kwargs) 
        self.tau = tau
        self.initializer = initializer
        
        # here the layers we want to subject to the threshold are specified
        targetlayers = [x for x in network.list_arguments() if x.startswith("conv") and x.endswith("weight")]
        self.targetlayers = targetlayers
        
        logging.info("RandomOut active on layers: %s" % self.targetlayers)
        
    def toc(self):
        for exe in self.exes:
            for array in exe.arg_arrays:
                array.wait_to_read()
        for exe in self.exes:
            for name, array in zip(exe._symbol.list_arguments(), exe.arg_arrays):
                self.queue.append((self.step, name, self.stat_func(array)))
                
        for exe in self.exes:
            weights = dict(zip(softmax.list_arguments(), exe.arg_arrays))
            grads = dict(zip(softmax.list_arguments(), exe.grad_arrays))
            numFilters = 0
            for name in self.targetlayers:
            
                filtersg = grads[name].asnumpy()
                filtersw = weights[name].asnumpy()

                #get random array to copy over
                filtersw_rand = mx.nd.array(filtersw.copy())
                self.initializer(name, filtersw_rand)
                filtersw_rand = filtersw_rand.asnumpy()
                
                agrads = [0.0] * len(filtersg)
                for i in range(len(filtersg)):
                    agrads[i] = np.absolute(filtersg[i]).sum()
                    if agrads[i] < self.tau:
                        numFilters = numFilters+1
                        #logging.info("RandomOut: filter %i of %s has been randomized because CGN=%f" % (i,name,agrads[i]))
                        filtersw[i] = filtersw_rand[i]

                #logging.info("%s, %s, %s" % (name, min(agrads),np.mean(agrads)))
            
                weights[name] = mx.nd.array(filtersw)
                #print filtersw
            if numFilters >0:
                #logging.info("numFilters replaced: %i"%numFilters)   
                exe.copy_params_from(arg_params=weights)
            
        self.activated = False
        return []
    



train_dataiter.reset()
if args.randomout == "True":
    model.fit(X=train_dataiter,
        eval_data=test_dataiter,
        eval_metric="accuracy",
        batch_end_callback=mx.callback.Speedometer(batch_size)
        ,monitor=RandomOutMonitor(initializer = model.initializer, network=softmax, tau=args.tau)
        )
else:
    model.fit(X=train_dataiter,
        eval_data=test_dataiter,
        eval_metric="accuracy",
        batch_end_callback=mx.callback.Speedometer(batch_size)
        )





