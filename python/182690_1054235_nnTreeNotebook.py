# importing utilities
import mxnet as mx
import numpy as np
import random
import re
import copy
from neuralAlgonometry import buildNNTree, encode_equations, EquationTree, readBlockJsonEquations,                              saveResults, updateWorksheet, dumpParams, writeJson, putNodeInTree, encodeTerminals
from nnTreeMain import lstmTreeInpOut, BucketEqIteratorInpOut, bucketIndex, precision, recall, Accuracy

# terminals:
variables = ['Symbol']
consts = ['NegativeOne', 'NaN', 'Infinity', 'Exp1', 'Pi', 'One',
          'Half', 'Integer', 'Rational', 'Float']
terminals = []
terminals.extend(variables) 
terminals.extend(consts)

intList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -2, -3]
ratList = ['2/5', '-1/2', '0']
floatList = [0.7]
varList = ['var_%d'%(d) for d in range(5)]

functionDictionary = encodeTerminals(terminals, intList, ratList, floatList, varList)
print "functionDictionary:", functionDictionary

# Model parameters and training setup 
params = None
contexts = mx.cpu(0)
num_hidden = 50
vocabSize = len(functionDictionary)
emb_dimension = 50
out_dimension = 1
batch_size = 1
splitRatio = 0.8 # proportion of equations in the train set
devSplit = 1
excludeFunc = ''
trainDepth = [1,2,3,4]
testDepth = [1,2,3,4]
dropout = 0.2
lr = 0.00001 # learning rate
mom = 0.7 # momentum
wd = 0.001 # weight decay
optimizer = "adam" # name of optimizer
num_epochs = 2 # number of training epochs
load_epoch = 0 # load pre-trained model starting from epoch number load_epoch
model_prefix = "notebookModel/model0/trained_model " # path to save model checkpoints
kv_store = "device" # KV_store 

# refer to section 1. for an explanation about different neural network classes
tTypeStr = 'lstmTreeInpOut' # neural network type. Other options: 'nnTreeInpOut', 'nnTree', 'lstmTree'
tType = lstmTreeInpOut  # neural network type. Other options: nnTreeInpOut, nnTree, lstmTree

# path to data: below is a smaller dataset that runs faster.
# file data/data4000_orig_inpuOut_with_neg.json is the data used in the paper
path = "data/data1000_depth4_inpuOut.json" 
result_path = "notebookModel/model0/res" # path for saving results such as train/test accuracies and other metrics

random.seed(1)
mx.random.seed(1)
np.random.seed(1)
[trainEquations, trainVars, _, trainLabels,
devEquations, devVars, _, devLabels,
testEquations, testVars, _, testLabels] \
          = readBlockJsonEquations(path, trainDepth=trainDepth, testDepth=testDepth,
                                   excludeFunc=excludeFunc, splitRatio=splitRatio, devSplit=devSplit, containsNumeric=True)
    
# uncomment if need to store the original data split
# writeJson(result_path+'trainData.json', trainEquations, ranges, trainVars, trainLabels, 6)
# writeJson(result_path+'devData.json', devEquations, ranges, devVars, devLabels, 6)
# writeJson(result_path+'testData.json', testEquations, ranges, testVars, testLabels, 6)

trainSamples = []
trainDataNames = []
for i, eq in enumerate(trainEquations):
    currNNTree = buildNNTree(treeType=tType , parsedEquation=eq, 
                                num_hidden=num_hidden, params=params, 
                                emb_dimension=emb_dimension, dropout=dropout)

    [currDataNames, _] = currNNTree.getDataNames(dataNames=[], nodeNumbers=[])
    trainDataNames.append(currDataNames)
    trainSamples.append(currNNTree)

testSamples = []
testDataNames = []
for i, eq in enumerate(testEquations):
    currNNTree = buildNNTree(treeType=tType , parsedEquation=eq, 
                                num_hidden=num_hidden, params=params, 
                                emb_dimension=emb_dimension, dropout=dropout)

    [currDataNames, _] = currNNTree.getDataNames(dataNames=[], nodeNumbers=[])
    testDataNames.append(currDataNames)
    testSamples.append(currNNTree)
    
# devSamples = []
# devDataNames = []
# for i, eq in enumerate(devEquations):
#     currNNTree = buildNNTree(treeType=tType , parsedEquation=eq, 
#                            num_hidden=num_hidden, params=params, 
#                            emb_dimension=emb_dimension, dropout=dropout)

#     [currDataNames, _] = currNNTree.getDataNames(dataNames=[], nodeNumbers=[])
#     devDataNames.append(currDataNames)
#     devSamples.append(currNNTree)

numTrainSamples = len(trainEquations)
trainBuckets = list(xrange(numTrainSamples))

numTestSamples = len(testEquations)
testBuckets = list(xrange(numTestSamples))

train_eq, _ = encode_equations(trainEquations, vocab=functionDictionary, invalid_label=-1, 
                                     invalid_key='\n', start_label=0)
data_train  = BucketEqIteratorInpOut(enEquations=train_eq, eqTreeList=trainSamples, batch_size=batch_size, 
                             buckets=trainBuckets, labels=trainLabels, vocabSize=len(functionDictionary),
                                    invalid_label=-1)

test_eq, _ = encode_equations(testEquations, vocab=functionDictionary, invalid_label=-1, 
                             invalid_key='\n', start_label=0)
data_test  = BucketEqIteratorInpOut(enEquations=test_eq, eqTreeList=testSamples, batch_size=batch_size, 
                             buckets=testBuckets, labels=testLabels, vocabSize=len(functionDictionary),
                                    invalid_label=-1, devFlag=1)


# numDevSamples = len(devEquations)
# devBuckets = list(xrange(numDevSamples))
# dev_eq, _ = encode_equations(devEquations, vocab=functionDictionary, invalid_label=-1, 
#                              invalid_key='\n', start_label=0)
# data_dev  = BucketEqIteratorInpOut(enEquations=dev_eq, eqTreeList=devSamples, batch_size=batch_size, 
#                              buckets=devBuckets, labels=devLabels, vocabSize=len(functionDictionary),
#                                     invalid_label=-1, devFlag=1)

cell = trainSamples[0]

def sym_gen(bucketIndexObj):

    label = mx.sym.Variable('softmax_label')

    bucketIDX = bucketIndexObj.bucketIDX
    devFlag = bucketIndexObj.devFlag

    if devFlag == 0:
        tree = trainSamples[bucketIDX]
    else:
        tree = testSamples[bucketIDX]


    [dataNames, nodeNumbers] = tree.getDataNames(dataNames=[], nodeNumbers=[])
    data_names_corr = [dataNames[i]+'_%d'%(nodeNumbers[i]) for i in range(len(dataNames))]
    nameDict = {}
    for i, dn in enumerate(dataNames):
        if dn not in nameDict:
            nameDict[dn+'_%d'%(nodeNumbers[i])] = mx.sym.Variable(name=dn+'_%d'%(nodeNumbers[i]))
        else:
            raise AssertionError("data name should not have been in the dictionary")

    if tType == lstmTreeInpOut:
        outputs, _ = tree.unroll(nameDict)
    else:
        outputs = tree.unroll(nameDict)

    pred = mx.sym.LogisticRegressionOutput(data=outputs, label=label, name='softmax')

    return pred, (data_names_corr), ('softmax_label',)

model = mx.mod.BucketingModule(
    sym_gen             = sym_gen,
    default_bucket_key  = bucketIndex(0, 0),
    context             = contexts,
    fixed_param_names  = [str(tTypeStr)+'_Equality_i2h_weight'])

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

acc = Accuracy()
prc = precision()
rcl = recall()

if load_epoch:
    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
        cell, model_prefix, load_epoch)
else:
    arg_params = None
    aux_params = None

opt_params = {
'learning_rate': lr,
'wd': wd
}

if optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
    opt_params['momentum'] = mom

model.fit(
train_data          = data_train,
eval_data           = data_test,
kvstore             = kv_store,
eval_metric         = [acc, prc, rcl],
optimizer           = optimizer,
optimizer_params    = opt_params,
initializer         = mx.init.Mixed([str(tTypeStr)+'_Equality_i2h_weight', '.*'], 
                                [mx.init.One(), mx.init.Xavier(factor_type="in", magnitude=2.34)]),
arg_params          = arg_params,
aux_params          = aux_params,
begin_epoch         = load_epoch,
num_epoch           = num_epochs,
epoch_end_callback  = mx.rnn.do_rnn_checkpoint(cell, model_prefix, 1) \
                                               if model_prefix else None)

accTrain = [acc.allVals[i] for i in range(0,len(acc.allVals),2)]
accVal   = [acc.allVals[i] for i in range(1,len(acc.allVals),2)]
prcTrain = [prc.allVals[i] for i in range(0,len(prc.allVals),2)]
prcVal   = [prc.allVals[i] for i in range(1,len(prc.allVals),2)]
rclTrain = [rcl.allVals[i] for i in range(0,len(rcl.allVals),2)]
rclVal   = [rcl.allVals[i] for i in range(1,len(rcl.allVals),2)]

trainMetrics = [accTrain, prcTrain, rclTrain]
valMetrics   = [accVal,     prcVal,   rclVal]

# args
if result_path:
    saveResults(result_path+'_train.json', {}, trainMetrics, valMetrics)
    trainPredicts = model.predict(data_train).asnumpy()
    np.save(result_path+'_train_predictions.npy', trainPredicts)
    with open(result_path+'_train_labels.txt', 'wa') as labelFile:
        for lbl in trainLabels:
            labelFile.write('{0}\n'.format(lbl.asscalar()))



