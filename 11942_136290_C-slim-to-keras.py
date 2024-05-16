import tensorflow as tf
import numpy as np

import os, sys
better_instructions = '2-CNN/4-ImageNet/4-ImageClassifier-inception_tf.ipynb'

if not os.path.isfile( '../models/tensorflow_zoo/models/README.md' ):
    print("Please follow the instructions in %s to get the Slim-Model-Zoo installed" % better_instructions)
else:
    sys.path.append('../models/tensorflow_zoo/models/slim')
    print("Model Zoo model code installed")

from datasets import dataset_utils

checkpoint_file = '../data/tensorflow_zoo/checkpoints/inception_v1.ckpt'
if not os.path.isfile( checkpoint_file ):
    print("Please follow the instructions in %s to get the Checkpoint installed" % better_instructions)
else:
    print("Checkpoint available locally")

if not os.path.isfile('../data/imagenet_synset_words.txt'):
    print("Please follow the instructions in %s to get the synset_words file" % better_instructions)
else:    
    print("ImageNet synset labels available")

slim = tf.contrib.slim
from nets import inception
#from preprocessing import inception_preprocessing

#image_size = inception.inception_v1.default_image_size
#image_size

tf.reset_default_graph()

if False:
    # Define the pre-processing chain within the graph - from a raw image
    input_image = tf.placeholder(tf.uint8, shape=[None, None, None, 3], name='input_image')
    processed_image = inception_preprocessing.preprocess_image(input_image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

processed_images = tf.placeholder(tf.float32, shape=[None, None, None, 3])

# Create the model - which uses the above pre-processing on image
#   it also uses the default arg scope to configure the batch norm parameters.
print("Model builder starting")

# Here is the actual model zoo model being instantiated :
with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits, end_points = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
#probabilities = tf.nn.softmax(logits)

# Create an operation that loads the pre-trained model from the checkpoint
init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, slim.get_model_variables('InceptionV1') )

print("Model defined")

capture_names =[] 
capture_values=dict()

# Now let's run the pre-trained model
with tf.Session() as sess:
    # This is the loader 'op' we defined above
    init_fn(sess)  
    
    #variables = tf.trainable_variables()
    variables = tf.model_variables()  # includes moving average information
    for variable in variables:
        name, value = variable.name, variable.eval()
        capture_names.append(name)
        capture_values[name] = value
        print("%20s %8d %s " % (value.shape, np.prod(value.shape), name, ))
        
    """
    BatchNorm variables are (beta,moving_mean,moving_variance) separately (trainable==beta only)
               (64,)       64 InceptionV1/Conv2d_1a_7x7/BatchNorm/beta:0 
               (64,)       64 InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean:0 
               (64,)       64 InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance:0     
    """

# This fixes a typo in the original slim library...
if 'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/weights:0' in capture_values:
    for w in ['weights:0', 'BatchNorm/beta:0', 'BatchNorm/moving_mean:0', 'BatchNorm/moving_variance:0']:
        capture_values['InceptionV1/Mixed_5b/Branch_2/Conv2d_0b_3x3/'+w] = (
            capture_values['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/'+w]
        )      

for e in sorted(end_points.keys()):
    print(e)

model_old = dict(names=capture_names, values=capture_values, 
                 check_names=end_points.keys(), check_values=dict())

import matplotlib.pyplot as plt
img_raw = plt.imread('../images/cat-with-tongue_224x224.jpg')
#img_raw.shape

# This is how the model_old does it in the pre-processing stages (so must be the same for model_new)
img = ( img_raw.astype('float32')/255.0 - 0.5 ) * 2.0
imgs = img[np.newaxis, :, :, :]

with tf.Session() as sess:
    # This is the loader 'op' we defined above
    init_fn(sess)  
    
    # This run grabs all the layer constants for the original photo image input
    check_names = model_old['check_names']
    end_points_values = sess.run([ end_points[k] for k in check_names ], feed_dict={processed_images: imgs})
    
    #model_old['check_values']={ k:end_points_values[i] for i,k in enumerate(check_names) }
    model_old['check_values']=dict( zip(check_names, end_points_values) )

# This is taken from https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image

# This is taken from https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              normalizer=True,
              activation='relu',
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution, `name + '_bn'` for the
            batch norm layer and `name + '_act'` for the
            activation layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name + '_act'
    else:
        conv_name = None
        bn_name = None
        act_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
            filters, (num_row, num_col),
            strides=strides, padding=padding,
            use_bias=False, name=conv_name)(x)
    if normalizer:
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation:
        x = Activation(activation, name=act_name)(x)
    return x

# Convenience function for 'standard' Inception concatenated blocks
def concatenated_block(x, specs, channel_axis, name):
    (br0, br1, br2, br3) = specs   # ((64,), (96,128), (16,32), (32,))
    
    branch_0 = conv2d_bn(x, br0[0], 1, 1, name=name+"_Branch_0_a_1x1")

    branch_1 = conv2d_bn(x, br1[0], 1, 1, name=name+"_Branch_1_a_1x1")
    branch_1 = conv2d_bn(branch_1, br1[1], 3, 3, name=name+"_Branch_1_b_3x3")

    branch_2 = conv2d_bn(x, br2[0], 1, 1, name=name+"_Branch_2_a_1x1")
    branch_2 = conv2d_bn(branch_2, br2[1], 3, 3, name=name+"_Branch_2_b_3x3")

    branch_3 = MaxPooling2D( (3, 3), strides=(1, 1), padding='same', name=name+"_Branch_3_a_max")(x)  
    branch_3 = conv2d_bn(branch_3, br3[0], 1, 1, name=name+"_Branch_3_b_1x1")

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name=name+"_Concatenated")
    return x

def InceptionV1(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """Instantiates the Inception v1 architecture.

    This architecture is defined in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 224x224.
    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns:
        A Keras model instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        #default_size=299,
        default_size=224,
        min_size=139,
        data_format=K.image_data_format(),
        include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # 'Sequential bit at start'
    x = img_input
    x = conv2d_bn(x,  64, 7, 7, strides=(2, 2), padding='same',  name='Conv2d_1a_7x7')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_2a_3x3')(x)  
    
    x = conv2d_bn(x,  64, 1, 1, strides=(1, 1), padding='same', name='Conv2d_2b_1x1')  
    x = conv2d_bn(x, 192, 3, 3, strides=(1, 1), padding='same', name='Conv2d_2c_3x3')  
    
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_3a_3x3')(x)  
    
    # Now the '3' level inception units
    x = concatenated_block(x, (( 64,), ( 96,128), (16, 32), ( 32,)), channel_axis, 'Mixed_3b')
    x = concatenated_block(x, ((128,), (128,192), (32, 96), ( 64,)), channel_axis, 'Mixed_3c')

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='MaxPool_4a_3x3')(x)  

    # Now the '4' level inception units
    x = concatenated_block(x, ((192,), ( 96,208), (16, 48), ( 64,)), channel_axis, 'Mixed_4b')
    x = concatenated_block(x, ((160,), (112,224), (24, 64), ( 64,)), channel_axis, 'Mixed_4c')
    x = concatenated_block(x, ((128,), (128,256), (24, 64), ( 64,)), channel_axis, 'Mixed_4d')
    x = concatenated_block(x, ((112,), (144,288), (32, 64), ( 64,)), channel_axis, 'Mixed_4e')
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_4f')

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='MaxPool_5a_2x2')(x)  

    # Now the '5' level inception units
    x = concatenated_block(x, ((256,), (160,320), (32,128), (128,)), channel_axis, 'Mixed_5b')
    x = concatenated_block(x, ((384,), (192,384), (48,128), (128,)), channel_axis, 'Mixed_5c')
    

    if include_top:
        # Classification block
        
        # 'AvgPool_0a_7x7'
        x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)  
        
        # 'Dropout_0b'
        x = Dropout(0.2)(x)  # slim has keep_prob (@0.8), keras uses drop_fraction
        
        #logits = conv2d_bn(x,  classes+1, 1, 1, strides=(1, 1), padding='valid', name='Logits',
        #                   normalizer=False, activation=None, )  
        
        # Write out the logits explictly, since it is pretty different
        x = Conv2D(classes+1, (1, 1), strides=(1,1), padding='valid', use_bias=True, name='Logits')(x)
        
        x = Flatten(name='Logits_flat')(x)
        #x = x[:, 1:]  # ??Shift up so that first class ('blank background') vanishes
        # Would be more efficient to strip off position[0] from the weights+bias terms directly in 'Logits'
        
        x = Activation('softmax', name='Predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='global_pooling')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(    name='global_pooling')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Finally : Create model
    model = Model(inputs, x, name='inception_v1')
    
    # LOAD model weights (TODO)
    
    return model

include_top=True

model_new = InceptionV1(weights='imagenet', include_top=include_top)

model_new.summary()
# 'Connected to' isn't showing up due to Keras bug https://github.com/fchollet/keras/issues/6286

def show_old_model_expected_shapes(model, name):
    v = model['values'][name]
    print('OLD    :', v.shape, name)

def show_new_model_expected_shapes(model, name):
    layer = model.get_layer(name)
    weights = layer.get_weights()
    for i, w in enumerate(weights):
        print('NEW[%d] : %s %s' % (i, w.shape, name))

# This depends on the naming conventions...
def copy_CNN_weight_with_bn(model_old, name_old, model_new, name_new):
    # See : https://github.com/fchollet/keras/issues/1671 
    weights = model_old['values'][name_old+'/weights:0']
    layer = model_new.get_layer(name_new+"_conv")
    layer.set_weights([weights])
    
    weights0 = model_old['values'][name_old+'/BatchNorm/beta:0']
    weights1 = model_old['values'][name_old+'/BatchNorm/moving_mean:0']
    weights2 = model_old['values'][name_old+'/BatchNorm/moving_variance:0']
    layer = model_new.get_layer(name_new+"_bn")
    weights_all = layer.get_weights()
    weights_all[0]=weights0
    weights_all[1]=weights1
    weights_all[2]=weights2
    layer.set_weights(weights_all)
    #print( weights_all[0] )
    #layer.set_weights([weights, np.zeros_like(weights), np.ones_like(weights), ])

show_old_model_expected_shapes(model_old, 'InceptionV1/Conv2d_1a_7x7/weights:0')
show_new_model_expected_shapes(model_new, 'Conv2d_1a_7x7_conv')

show_old_model_expected_shapes(model_old, 'InceptionV1/Conv2d_1a_7x7/BatchNorm/beta:0')
show_old_model_expected_shapes(model_old, 'InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean:0')
show_old_model_expected_shapes(model_old, 'InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance:0')
show_new_model_expected_shapes(model_new, 'Conv2d_1a_7x7_bn')

#model_old['values']['InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_mean:0']
#model_old['values']['InceptionV1/Conv2d_1a_7x7/BatchNorm/moving_variance:0']

#copy_CNN_weight_with_bn(model_old, 'InceptionV1/Conv2d_1a_7x7', model_new, 'Conv2d_1a_7x7')
#copy_CNN_weight_with_bn(model_old, 'InceptionV1/Conv2d_2b_1x1', model_new, 'Conv2d_2b_1x1')
#copy_CNN_weight_with_bn(model_old, 'InceptionV1/Conv2d_2c_3x3', model_new, 'Conv2d_2c_3x3')
for block in [
        'Conv2d_1a_7x7', 
        'Conv2d_2b_1x1',
        'Conv2d_2c_3x3',
    ]:
    print("Copying %s" % (block,))    
    copy_CNN_weight_with_bn(model_old, 'InceptionV1/'+block, model_new, block)

print("Finished All")

# This depends on the naming conventions...
def copy_inception_block_weights(model_old, block_old, model_new, block_new):
    # e.g. FROM : InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1
    #        TO : Mixed_3b_Branch_1_a_1x1
    # block_old = 'InceptionV1/Mixed_3b'
    # block_new = 'Mixed_3b'
    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_0/Conv2d_0a_1x1', model_new, block_new+'_Branch_0_a_1x1')
    
    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_1/Conv2d_0a_1x1', model_new, block_new+'_Branch_1_a_1x1')
    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_1/Conv2d_0b_3x3', model_new, block_new+'_Branch_1_b_3x3')

    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_2/Conv2d_0a_1x1', model_new, block_new+'_Branch_2_a_1x1')
    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_2/Conv2d_0b_3x3', model_new, block_new+'_Branch_2_b_3x3')

    copy_CNN_weight_with_bn(model_old, block_old+'/Branch_3/Conv2d_0b_1x1', model_new, block_new+'_Branch_3_b_1x1')

#copy_inception_block_weights(model_old, 'InceptionV1/Mixed_3b', model_new, 'Mixed_3b')
#copy_inception_block_weights(model_old, 'InceptionV1/Mixed_3c', model_new, 'Mixed_3c')

for block in [
        'Mixed_3b', 'Mixed_3c', 
        'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 
        'Mixed_5b', 'Mixed_5c', 
    ]:
    print("Copying %s" % (block,))    
    copy_inception_block_weights(model_old, 'InceptionV1/'+block, model_new, block)

print("Finished All")

# This depends on the naming conventions...
def copy_CNN_weight_with_bias(model_old, name_old, model_new, name_new):
    weights0 = model_old['values'][name_old+'/weights:0']
    weights1 = model_old['values'][name_old+'/biases:0']
    layer = model_new.get_layer(name_new)
    weights_all = layer.get_weights()
    weights_all[0]=weights0
    weights_all[1]=weights1
    layer.set_weights(weights_all)

if include_top:
    show_old_model_expected_shapes(model_old, 'InceptionV1/Logits/Conv2d_0c_1x1/weights:0')
    show_old_model_expected_shapes(model_old, 'InceptionV1/Logits/Conv2d_0c_1x1/biases:0')
    show_new_model_expected_shapes(model_new, 'Logits')

    print("Copying Logits")
    copy_CNN_weight_with_bias(model_old, 'InceptionV1/Logits/Conv2d_0c_1x1', model_new, 'Logits')
print("Finished All")

imgs.shape

def check_image_outputs(model_old, name_old, model_new, name_new, images, idx):
    images_old = model_old['check_values'][name_old]
    print("OLD :", images_old.shape, np.min(images_old), np.max(images_old) )
    
    # See : http://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    output_layer = model_new.get_layer(name_new)
    get_check_value = K.function([model_new.input, K.learning_phase()], 
                                 [output_layer.output,])
    check_value = get_check_value([images, 0])  # '0' is for 'learning_phase'
    images_new = check_value[0]
    print("NEW :", images_new.shape, np.min(images_new), np.max(images_new) )
    
    total_diff = np.sum( np.abs(images_new - images_old) )
    print("total_diff =", total_diff)
    
    if len(images_old.shape)<4: return
    
    def no_axes():
        plt.gca().xaxis.set_visible(False)    
        plt.gca().yaxis.set_visible(False)    
        
    plt.figure(figsize=(9,4))

    # https://matplotlib.org/examples/color/colormaps_reference.html  bwr(+/-) or Blues(0+)
    plt.subplot2grid( (1,2), (0,0) ); no_axes()
    plt.imshow(images_old[0, :,:,idx], cmap='Blues', vmin=0.) # , vmin=0. , vmax=1.
    plt.subplot2grid( (1,2), (0,1) ); no_axes()
    plt.imshow(images_new[0, :,:,idx], cmap='Blues', vmin=0.) # ,vmin=-1., vmax=1.

    #plt.tight_layout()    
    plt.show()

# These should show attractive, identical images on left and right sides
#check_image_outputs(model_old, 'Conv2d_1a_7x7', model_new, 'Conv2d_1a_7x7_act', imgs, 31)
#check_image_outputs(model_old, 'MaxPool_2a_3x3', model_new, 'MaxPool_2a_3x3', imgs, 11)
#check_image_outputs(model_old, 'Conv2d_2b_1x1', model_new, 'Conv2d_2b_1x1_act', imgs, 5)
#check_image_outputs(model_old, 'Conv2d_2c_3x3', model_new, 'Conv2d_2c_3x3_act', imgs, 35)
check_image_outputs(model_old, 'MaxPool_3a_3x3', model_new, 'MaxPool_3a_3x3', imgs, 5)
#check_image_outputs(model_old, 'Mixed_3b', model_new, 'Mixed_3b_Concatenated', imgs, 25)
#check_image_outputs(model_old, 'Mixed_3c', model_new, 'Mixed_3c_Concatenated', imgs, 25)
#check_image_outputs(model_old, 'MaxPool_4a_3x3', model_new, 'MaxPool_4a_3x3', imgs, 25)
#check_image_outputs(model_old, 'MaxPool_5a_2x2', model_new, 'MaxPool_5a_2x2', imgs, 25)

if include_top:
    # No images for these ones...
    #check_image_outputs(model_old, 'Logits', model_new, 'Logits_flat', imgs, -1)
    check_image_outputs(model_old, 'Predictions', model_new, 'Predictions', imgs, -1)

if include_top:
    model_file = 'inception_v1_weights_tf_dim_ordering_tf_kernels.h5'
else:
    model_file = 'inception_v1_weights_tf_dim_ordering_tf_kernels_notop.h5'

# This assumes that the model_weights will be loaded back into the same structure
model_new.save_weights(model_file)



