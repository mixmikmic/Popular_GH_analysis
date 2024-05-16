from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.summary()

# pip install pydot-ng
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

from keras.applications.vgg16 import VGG16

model = VGG16(weights=None, include_top=False)

SVG(model_to_dot(model).create(prog='dot', format='svg'))

from keras.applications.resnet50 import ResNet50

model = ResNet50(weights=None)

SVG(model_to_dot(model).create(prog='dot', format='svg'))

# All these models also come with these two functions
from keras.applications.resnet50 import preprocess_input, decode_predictions

# Make sure to use them when making predictions!!!



