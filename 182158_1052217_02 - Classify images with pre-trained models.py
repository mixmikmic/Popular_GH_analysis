get_ipython().system('wget http://data.dmlc.ml/models/imagenet/vgg/vgg16-symbol.json -O vgg16-symbol.json')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/vgg/vgg16-0000.params -O vgg16-0000.params')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-symbol.json -O Inception-BN-symbol.json')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-0126.params -O Inception-BN-0000.params')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-symbol.json -O resnet-152-symbol.json')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-0000.params -O resnet-152-0000.params')
get_ipython().system('wget http://data.dmlc.ml/models/imagenet/synset.txt -O synset.txt')

get_ipython().system('head -48 vgg16-symbol.json')

get_ipython().system('head -10 synset.txt')

import mxnet as mx
import numpy as np
import cv2,sys,time
from collections import namedtuple
from IPython.core.display import Image, display

print("MXNet version: %s" % mx.__version__)

def loadModel(modelname, gpu=False):
        sym, arg_params, aux_params = mx.model.load_checkpoint(modelname, 0)
        arg_params['prob_label'] = mx.nd.array([0])
        arg_params['softmax_label'] = mx.nd.array([0])
        if gpu:
            mod = mx.mod.Module(symbol=sym, context=mx.gpu(0))
        else:
            mod = mx.mod.Module(symbol=sym)
        mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
        mod.set_params(arg_params, aux_params)
        return mod

def loadCategories():
        synsetfile = open('synset.txt', 'r')
        synsets = []
        for l in synsetfile:
                synsets.append(l.rstrip())
        return synsets
    
synsets = loadCategories()
print(synsets[:10])

def prepareNDArray(filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224,))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        array = mx.nd.array(img)
        print(array.shape)
        return array

def predict(filename, model, categories, n):
        array = prepareNDArray(filename)
        Batch = namedtuple('Batch', ['data'])
        t1 = time.time()
        model.forward(Batch([array]))
        prob = model.get_outputs()[0].asnumpy()
        t2 = time.time()
        print("Predicted in %.2f microseconds" % (t2-t1))
        prob = np.squeeze(prob)
        sortedprobindex = np.argsort(prob)[::-1]
        
        topn = []
        for i in sortedprobindex[0:n]:
                topn.append((prob[i], categories[i]))
        return topn

def init(modelname, gpu=False):
        model = loadModel(modelname,gpu)
        categories = loadCategories()
        return model, categories

vgg16,categories = init("vgg16")
resnet152,categories = init("resnet-152")
inceptionv3,categories = init("Inception-BN")

params = vgg16.get_params()

layers = []
for layer in params[0].keys():
    layers.append(layer)
    
layers.sort()    
print(layers)

print(params[0]['fc8_weight'])

image = "violin.jpg"

display(Image(filename=image))

topn = 5
print ("*** VGG16")
print (predict(image,vgg16,categories,topn))
print ("*** ResNet-152")
print (predict(image,resnet152,categories,topn))
print ("*** Inception v3")
print (predict(image,inceptionv3,categories,topn))

vgg16,categories = init("vgg16", gpu=True)
resnet152,categories = init("resnet-152", gpu=True)
inceptionv3,categories = init("Inception-BN", gpu=True)

print ("*** VGG16")
print (predict(image,vgg16,categories,topn))
print ("*** ResNet-152")
print (predict(image,resnet152,categories,topn))
print ("*** Inception v3")
print (predict(image,inceptionv3,categories,topn))



