import sys
sys.path.append('../code/')

from inception_resnet_v1 import *
model = InceptionResNetV1(weights_path='../model/keras/weights/facenet_keras_weights.h5')

from coremltools.proto import NeuralNetwork_pb2

# The conversion function for Lambda layers.
def convert_lambda(layer):
    if layer.function == scaling:
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "scaling"
        return params
    else:
        return None

import coremltools

coreml_model = coremltools.converters.keras.convert(
    model,
    input_names="image",
    image_input_names="image",
    output_names="output",
    add_custom_layers=True,
    custom_conversion_functions={ "Lambda": convert_lambda })



