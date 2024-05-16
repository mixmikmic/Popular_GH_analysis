import mxnet as mx
import numpy as np
import cv2, time
from IPython.display import Image
from collections import namedtuple

np.set_printoptions(precision=4, suppress=True)

def loadImage(filename):
  img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  img = img / 255
  img = np.expand_dims(img, axis=0)
  img = np.expand_dims(img, axis=0)
  return mx.nd.array(img)

def predict(model, filename):
  array = loadImage(filename)
  #print(array.shape)
  Batch = namedtuple('Batch', ['data'])
  mod.forward(Batch([array]))
  pred = mod.get_outputs()[0].asnumpy()
  return pred

def loadModel(model, epochs):
  model, arg_params, aux_params = mx.model.load_checkpoint(model, epochs)
  mod = mx.mod.Module(model)
  mod.bind(for_training=False, data_shapes=[('data', (1,1,28,28))])
  mod.set_params(arg_params, aux_params)
  return mod

#mod = loadModel("mlp", 50)
mod = loadModel("lenet", 25)

Image(filename="./0.png")

print(predict(mod, "./0.png"))

Image(filename="./1.png")

print(predict(mod, "./1.png"))

Image(filename="./2.png")

print(predict(mod, "./2.png"))

Image(filename="./3.png")

print(predict(mod, "./3.png"))

Image(filename="./4.png")

print(predict(mod, "./4.png"))

Image(filename="./5.png")

print(predict(mod, "./5.png"))

Image(filename="./6.png")

print(predict(mod, "./6.png"))

Image(filename="./7.png")

print(predict(mod, "./7.png"))

Image(filename="./8.png")

print(predict(mod, "./8.png"))

Image(filename="./9.png")

print(predict(mod, "./9.png"))



