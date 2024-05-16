import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import time

ver = tf.__version__
# e.g., ver = "0.9.0" or "0.10.0rc0" 
ver_parts = ver.split(".")
ver_main = int(ver_parts[0])
ver_sub = int(ver_parts[1])
# r0.10 was released on 8/22/16. However, this does not contain the latest version of slim.
# For that, we need to check if you have the latest snapshot. One simple test is to see if the
# resnet has been imported...
try:
    import tensorflow.contrib.slim.nets
    mynet = slim.nets.resnet_v1
    print 'Pass: You have an up to date snapshot of TF version {}.{}'.format(ver_main, ver_sub)
except ImportError, e:
    print 'Fail: You have TF version {}.{}, you need latest snapshot'.format(ver_main, ver_sub)

def regression_model(inputs, is_training=True, scope="deep_regression"):
  """Creates the regression model.
  
  Args:
    input_node: A node that yields a `Tensor` of size [batch_size, dimensions].
    is_training: Whether or not we're currently training the model.
    scope: An optional variable_op scope for the model.
  
  Returns:
    output_node: 1-D `Tensor` of shape [batch_size] of responses.
    nodes: A dict of nodes representing the hidden layers.
  """
  with tf.variable_op_scope([input_node], scope):
    nodes = {}
    # Set the default weight _regularizer and acvitation for each fully_connected layer.
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(0.01)):
      
      # Creates a fully connected layer from the inputs with 10 hidden units.
      fc1_node = slim.fully_connected(inputs, 10, scope='fc1')
      nodes['fc1'] = fc1_node
        
      # Adds a dropout layer to prevent over-fitting.
      dropout_node = slim.dropout(fc1_node, 0.8, is_training=is_training)
      
      # Adds another fully connected layer with 5 hidden units.
      fc2_node = slim.fully_connected(dropout_node, 5, scope='fc2')
      nodes['fc2'] = fc2_node
      
      # Creates a fully-connected layer with a single hidden unit. Note that the
      # layer is made linear by setting activation_fn=None.
      prediction_node = slim.fully_connected(fc2_node, 1, activation_fn=None, scope='prediction')
      nodes['out'] = prediction_node

      return prediction_node, nodes

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
  # Dummy placeholders for arbitrary number of 1d inputs and outputs
  input_node = tf.placeholder(tf.float32, shape=(None, 1))
  output_node = tf.placeholder(tf.float32, shape=(None, 1))
  
  # Build model
  prediction_node, all_nodes = regression_model(input_node)
 
  # Print name and shape of each tensor.
  print "Layers"
  for k, v in all_nodes.iteritems():
    print 'name = {}, shape = {}'.format(v.name, v.get_shape())
    
  # Print name and shape of parameter nodes  (values not yet initialized)
  print "Parameters"
  for v in slim.get_model_variables():
    print 'name = {}, shape = {}'.format(v.name, v.get_shape())
       

def produce_batch(batch_size, noise=0.3):
  xs = np.random.random(size=[batch_size, 1]) * 10
  ys = np.sin(xs) + 5 + np.random.normal(size=[batch_size, 1], scale=noise)
  return [xs.astype(np.float32), ys.astype(np.float32)]

x_train, y_train = produce_batch(100)
x_test, y_test = produce_batch(100)
plt.scatter(x_train, y_train)

# Everytime we run training, we need to store the model checkpoint in a new directory,
# in case anything has changed.
import time
ts = time.time()
ckpt_dir = '/tmp/tf/regression_model/model{}'.format(ts) # Place to store the checkpoint.
print('Saving to {}'.format(ckpt_dir))

def convert_data_to_tensors(x, y):
  input_tensor = tf.constant(x)
  input_tensor.set_shape([None, 1])
  output_tensor = tf.constant(y)
  output_tensor.set_shape([None, 1])
  return input_tensor, output_tensor

graph = tf.Graph() # new graph
with graph.as_default():
  input_node, output_node = convert_data_to_tensors(x_train, y_train)

  # Make the model.
  prediction_node, nodes = regression_model(input_node, is_training=True)
 
  # Add the loss function to the graph.
  loss_node = slim.losses.sum_of_squares(prediction_node, output_node)
  # The total loss is the uers's loss plus any regularization losses.
  total_loss_node = slim.losses.get_total_loss()

  # Create some summaries to visualize the training process:
  ## TODO: add summaries.py to 3p
  #slim.summaries.add_scalar_summary(total_loss, 'Total Loss', print_summary=True)
  
  # Specify the optimizer and create the train op:
  optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
  train_op_node = slim.learning.create_train_op(total_loss_node, optimizer) 

  # Run the training inside a session.
  final_loss = slim.learning.train(
    train_op_node,
    logdir=ckpt_dir,
    number_of_steps=500,
    save_summaries_secs=1)
  
print("Finished training. Last batch loss:", final_loss)
print("Checkpoint saved in %s" % ckpt_dir)

graph = tf.Graph()  # Make a new graph
with graph.as_default():
    input_node, output_node = convert_data_to_tensors(x_train, y_train)
    prediction_node, nodes = regression_model(input_node, is_training=True)

    # Add multiple loss nodes.
    sum_of_squares_loss_node = slim.losses.sum_of_squares(prediction_node, output_node)
    absolute_difference_loss_node = slim.losses.absolute_difference(prediction_node, output_node)

    # The following two ways to compute the total loss are equivalent
    regularization_loss_node = tf.add_n(slim.losses.get_regularization_losses())
    total_loss1_node = sum_of_squares_loss_node + absolute_difference_loss_node + regularization_loss_node

    # Regularization Loss is included in the total loss by default.
    # This is good for training, but not for testing.
    total_loss2_node = slim.losses.get_total_loss(add_regularization_losses=True)
    
    init_node = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_node) # Will randomize the parameters.
        total_loss1, total_loss2 = sess.run([total_loss1_node, total_loss2_node])
        print('Total Loss1: %f' % total_loss1)
        print('Total Loss2: %f' % total_loss2)

        print('Regularization Losses:')
        for loss_node in slim.losses.get_regularization_losses():
            print(loss_node)

        print('Loss Functions:')
        for loss_node in slim.losses.get_losses():
            print(loss_node)

with tf.Graph().as_default():
    input_node, output_node = convert_data_to_tensors(x_test, y_test)
  
    # Create the model structure. (Parameters will be loaded below.)
    prediction_node, nodes = regression_model(input_node, is_training=False)

    # Make a session which restores the old parameters from a checkpoint.
    sv = tf.train.Supervisor(logdir=ckpt_dir)
    with sv.managed_session() as sess:
        inputs, predictions, true_outputs = sess.run([input_node, prediction_node, output_node])

plt.scatter(inputs, true_outputs, c='r');
plt.scatter(inputs, predictions, c='b');
plt.title('red=true, blue=predicted')

with tf.Graph().as_default():
    input_node = tf.placeholder(tf.float32, shape=(None, 1))
    output_node = tf.placeholder(tf.float32, shape=(None, 1))
    prediction_node, nodes = regression_model(input_node, is_training=False)
  
    sv = tf.train.Supervisor(logdir=ckpt_dir)
    with sv.managed_session() as sess:
        model_variables = slim.get_model_variables()
        for v in model_variables:
            val = sess.run(v)
            print v.name, val.shape, val

with tf.Graph().as_default():
    input_node, output_node = convert_data_to_tensors(x_test, y_test)
    prediction_node, nodes = regression_model(input_node, is_training=False)

    # Specify metrics to evaluate:
    names_to_value_nodes, names_to_update_nodes = slim.metrics.aggregate_metric_map({
      'Mean Squared Error': slim.metrics.streaming_mean_squared_error(prediction_node, output_node),
      'Mean Absolute Error': slim.metrics.streaming_mean_absolute_error(prediction_node, output_node)
    })


    init_node = tf.group(
        tf.initialize_all_variables(),
        tf.initialize_local_variables())

    # Make a session which restores the old graph parameters, and then run eval.
    sv = tf.train.Supervisor(logdir=ckpt_dir)
    with sv.managed_session() as sess:
        metric_values = slim.evaluation.evaluation(
            sess,
            num_evals=1, # Single pass over data
            init_op=init_node,
            eval_op=names_to_update_nodes.values(),
            final_op=names_to_value_nodes.values())

    names_to_values = dict(zip(names_to_value_nodes.keys(), metric_values))
    for key, value in names_to_values.iteritems():
      print('%s: %f' % (key, value))

# Retrieve the data.
import six
import sys
from six.moves import urllib
import os
url = 'https://github.com/probml/pyprobml/blob/master/tensorflow/cifar10_test.tfrecord'
cifar10_folder = '/tmp/tf/cifar10'
filename = os.path.join(cifar10_folder, 'cifar10_test.tfrecord')
#urllib.request.urlretrieve(url, filename) # Corrupted?
cifar10_folder = '/tmp/cifar10' # Data put here using Nathan's script

# %load https://raw.githubusercontent.com/probml/pyprobml/master/tensorflow/cifar10_make_slim_dataset.py
import tensorflow as tf

 
def make_cifar_dataset(split_name, tf_folder):
    """Make a dataset object from cifar10 tfrecord file.

    Args:
      split_name: "train" or "test"
      tf_folder: The base directory of the dataset sources.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}

    ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [32 x 32 x 3] color image.',
        'label': 'A single integer between 0 and 9',
    }
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    file_pattern =  '%s/cifar10_%s.tfrecord' % (tf_folder, split_name)

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[32, 32, 3]),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
   
    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=ITEMS_TO_DESCRIPTIONS)

with tf.Graph().as_default(): 
    dataset = make_cifar_dataset('test', cifar10_folder) # Must make dataset in same graph as data_provider and model
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32, common_queue_min=1)
    print("This dataset contains data of type {}".format(data_provider.list_items()))
    
    image_node, label_node = data_provider.get(['image', 'label'])
    with tf.Session() as sess:    
        with slim.queues.QueueRunners(sess):
            for i in xrange(4):
                image, label = sess.run([image_node, label_node])
                plt.figure()
                plt.imshow(image)
                plt.title(label)
                plt.axis('off')
                plt.show()

def my_cnn(images, num_classes, is_training):  # is_training is not used...
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(images, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 192)
        net = slim.fully_connected(net, num_classes, activation_fn=None)       
        return net

with tf.Graph().as_default():
    # The model can handle any input size because the first layer is convolutional.
    # The size of the model is determined when image_node is passed into the Model function.
    # All images must be the same size, because of the fully connected layers, which require a fixed size input.
    n_images = 3
    image_node = tf.random_uniform([n_images, 28, 28, 3], maxval=1)
    
    # Create the model.
    logits_node = my_cnn(image_node, 10, True)
    prob_node = tf.nn.softmax(logits_node)
  
    # Initialize all the variables (including parameters) randomly.
    init_op = tf.initialize_all_variables()
  
    with tf.Session() as sess:
        # Run the init_op, evaluate the model outputs and print the results:
        sess.run(init_op)
        probs = sess.run(prob_node)
        
print(probs.shape)  # 3x10
print(probs)
print(np.sum(probs, 1)) # Each row sums to 1

def preprocess(image, is_training):
    """Preprocesses the given image.

    Args:
        image: An image `Tensor` of size [32, 32, 3].
        is_training: A boolean, whether or not we're in training mode.

    Returns:
        A preprocessed and cropped image of shape [24, 24, 3]
    """
    height = 24
    width = 24

    if is_training:
        # Randomly crop a [height, width] section of the image.
        image = tf.random_crop(image, [height, width, 3])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomize the pixel values.
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    else:
        # Crop the central [height, width] of the image.
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_whitening(image)

def create_cifar_tensors(split_name='test'):
    dataset = make_cifar_dataset(split_name, cifar10_folder) 
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image, label = data_provider.get(['image', 'label'])
    
    # We have to use the preprocess function, otherwise we get this error:
    # TypeError: Expected uint8, got -0.059850560166457976 of type 'float' instead.
    image = preprocess(image, is_training=True)

    # Batch it up.
    images, labels = tf.train.batch(
          [image, label],
          batch_size=BATCH_SIZE,
          num_threads=2,
          capacity=10 * BATCH_SIZE)
    
    return images, labels

# Train the model on some labeled data.

CHECKPOINT_DIR = '/tmp/tf/cifar10_model/model{}'.format(time.time()) 
print('Saving model to {}'.format(CHECKPOINT_DIR))
BATCH_SIZE = 64
NUM_CLASSES = 10

with tf.Graph().as_default():
    # We train on the test set, just because it's smaller.
    images, labels = create_cifar_tensors('test')
  
    # Create the model:
    logits = my_cnn(images, num_classes=NUM_CLASSES, is_training=True)
 
    # Specify the loss function:
    one_hot_labels = slim.one_hot_encoding(labels, 10)
    slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    # Create some summaries to visualize the training process:
    # TODO: Make this be printed to stdout during training.
    tf.scalar_summary('losses/Total Loss', total_loss)
  
    # Specify the optimizer and create the train op:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Run the training:
    final_loss = slim.learning.train(
      train_op,
      logdir=CHECKPOINT_DIR,
      number_of_steps=50,
      save_summaries_secs=1)
  
    print('Finished training. Last batch loss {}'.format(final_loss))

import math
BATCH_SIZE = 64
EVAL_DIR = CHECKPOINT_DIR
NUM_TEST_IMAGES = 10000

with tf.Graph().as_default():
    images, labels = create_cifar_tensors('test')
    logits = my_cnn(images, num_classes=NUM_CLASSES, is_training=False)
    predictions = tf.argmax(logits, 1)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'eval/Recall@5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
    })
    
    # Create the summary ops such that they also print out to std output:
    summary_ops = []
    for name, value in names_to_values.iteritems():
        op = tf.scalar_summary(name, value, collections=[])
        op = tf.Print(op, [value], name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        summary_ops.append(op)

    
    # This ensures that we make a single pass over all of the data.
    # TODO: How to handle case where test size is not integer multiple of batch size?
    num_batches = math.ceil(NUM_TEST_IMAGES / float(BATCH_SIZE))

    print('Running evaluation Loop...')
    
    slim.evaluation.evaluation_loop(
        master='',
        checkpoint_dir=CHECKPOINT_DIR,  # Restores model from this checkpoint.
        logdir=EVAL_DIR,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        summary_op=tf.merge_summary(summary_ops),
        max_number_of_evaluations=1, # Only run loop once
        eval_interval_secs=1)
    print('Done!')
    
# To print the eval, we need to use tensorboard to decode the files in EVAL_DIR...
# TODO: Make this print to screen while running.

with tf.Graph().as_default():
    # Use random images. They must be the same size as the preprocessed training set, because we are restoring
    # fully connected layers from the checkpoint.
    n_images = 5
    image_node = tf.random_uniform([n_images, 24, 24, 3], maxval=1)
  
    # Create the model structure. (Parameters will be loaded below.)
    logits_node = my_cnn(image_node, num_classes=NUM_CLASSES, is_training=False)
    prob_node = tf.nn.softmax(logits_node)
    
    # Make a session which restores the old parameters from a checkpoint.
    sv = tf.train.Supervisor(logdir=CHECKPOINT_DIR)
    with sv.managed_session() as sess:
        probs = sess.run(prob_node)

print(probs.shape)  # 5x10
print(probs)
print(np.sum(probs, 1)) # Each row sums to 1

import six
import sys
from six.moves import urllib
import os
import tarfile

#url = 'http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz'
url = "http://download.tensorflow.org/models/inception_v3.tar.gz"
inception_folder = '/tmp/tf/inception-v3'
if not os.path.exists(inception_folder):
    os.mkdir(inception_folder)
filename = url.split('/')[-1]
filepath = os.path.join(inception_folder, filename)
if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(inception_folder)

import tensorflow.contrib.slim.nets as nets
with tf.Graph().as_default():
    # Create data source
    #image_node = tf.ones((32, 224, 224, 3))
    dataset = make_cifar_dataset('test', cifar10_folder) 
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32, common_queue_min=1)
    image_node, label_node = data_provider.get(['image', 'label'])
    # Create model. 
    scope = nets.inception.inception_v3_arg_scope(is_training=True)
    with slim.arg_scope(scope):
        logits_node, layer_nodes = nets.inception.inception_v3(image_node, num_classes=10)
    # Apply model to data
    with tf.Session() as sess:    
        with slim.queues.QueueRunners(sess):
            for i in xrange(4):
                image, label = sess.run([image_node, label_node])
                logits = sess.run([logits_node])
                predicted_label = np.argmax(logits)
                plt.figure()
                plt.subplot(1, 1, i)
                plt.imshow(image)
                plt.title('true label {}, predicted {}'.format(label, predicted_label))
                plt.axis('off')
                plt.show()



