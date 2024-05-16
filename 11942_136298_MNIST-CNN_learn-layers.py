"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

#  FROM : https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier
#  CODE : https://www.tensorflow.org/code/tensorflow/examples/tutorials/layers/cnn_mnist.py

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pickle

import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)  # Quite a lot...
#tf.logging.set_verbosity(tf.logging.WARN)   # This prevents Logging ...

do_training = False

import sys
print(sys.version)
print('Tensorflow:',tf.__version__)

def cnn_model_fn(features, integer_labels, mode):
  """Model function for CNN."""
  #print("Run cnn_model_fn, mode=%s" % (mode,))

  if type(features) is dict:
    #print("New-style feature input")
    features_images=features['images']
  else:
    print("OLD-style feature input (DEPRECATED)")
    features_images=features

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features_images, [-1, 28, 28, 1], name='input_layer')

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training= (mode == learn.ModeKeys.TRAIN) )

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  #logits = tf.Print(logits, [input_layer.get_shape(), integer_labels.get_shape()], "Debug size information : ", first_n=1)
  #logits = tf.layers.dense(inputs=dense, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(integer_labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=onehot_labels)
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=[ cls_targets[0] ])

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.001,
      #optimizer="SGD")
      optimizer="Adam")

  # Generate Predictions
  predictions = {
    "classes":       tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor"), 
    "logits":        logits,
    #"before_and_after":( input_layer, logits ),
    #"before_and_after":dict(input_layer=input_layer, logits=logits),
  }
    
  # For OLD-STYLE inputs (needs wierd 'evaluate' metric)
  if mode == model_fn_lib.ModeKeys.EVAL:  
    predictions['input_grad'] = tf.gradients(loss, [input_layer])[0]
    
  # For NEW-STYLE inputs (can smuggle in extra parameters)
  if type(features) is dict and 'fake_targets' in features: 
    loss_vs_target = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, 
        labels=features['fake_targets']
    )
    predictions['image_gradient_vs_fake_target'] = tf.gradients(loss_vs_target, [input_layer])[0]

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

# Create the Estimator : https://www.tensorflow.org/extend/estimators
mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="mnist_model/cnn")  # This is relative to the ipynb

# Check : the checkpoints file in 'mnist_model/cnn' has filenames that are in same directory

if False:
    print( mnist_classifier.get_variable_names() )
    #mnist_classifier.get_variable_value('conv2d/bias')

    #mnist_classifier.save()

    #tf.get_variable('input_layer')
    print( tf.global_variables() )
    print( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) )
    print( [n.name for n in tf.get_default_graph().as_graph_def().node] )

# Load training and eval data
mnist = learn.datasets.load_dataset("mnist")

train_data   = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

eval_data    = mnist.test.images  # Returns np.array
eval_labels  = np.asarray(mnist.test.labels, dtype=np.int32)

#print(eval_labels[7])
print("Data Loaded")


#https://www.tensorflow.org/get_started/input_fn#passing_input_fn_data_to_your_model
def mnist_batch_input_fn(dataset, batch_size=100, seed=None, num_epochs=1):  
    # If seed is defined, this will shuffle data into batches
    
    if False:  # This is the idea (but numpy, rather than Tensors)
        feature_dict = dict( images = dataset.images )
        labels       = np.asarray( dataset.labels, dtype=np.int32)
        return feature_dict, labels # but batch_size==EVERYTHING_AT_ONCE, unless we batch it up...
        
    np_labels = np.asarray( dataset.labels, dtype=np.int32)
    
    # Instead, build a Tensor dict 
    all_images = tf.constant( dataset.images, shape=dataset.images.shape, verify_shape=True )
    all_labels = tf.constant( np_labels,      shape=np_labels.shape, verify_shape=True )

    print("mnist_batch_input_fn sizing : ", 
          dataset.images.shape, 
          np.asarray( dataset.labels, dtype=np.int32).shape, 
          np.asarray( [dataset.labels], dtype=np.int32).T.shape,
         )
    
    # And create a 'feeder' to batch up the data appropriately...
    image, label = tf.train.slice_input_producer( [all_images, all_labels], 
                                                  num_epochs=num_epochs,
                                                  shuffle=(seed is not None), seed=seed,
                                                )
    
    dataset_dict = dict( images=image, labels=label ) # This becomes pluralized into batches by .batch()
    
    batch_dict = tf.train.batch( dataset_dict, batch_size,
                                num_threads=1, capacity=batch_size*2, 
                                enqueue_many=False, shapes=None, dynamic_pad=False, 
                                allow_smaller_final_batch=False, 
                                shared_name=None, name=None)
    
    
    batch_labels = batch_dict.pop('labels')
    
    # Return : 
    # 1) a mapping of feature columns to Tensors with the corresponding feature data, and 
    # 2) a Tensor containing labels
    return batch_dict, batch_labels

batch_size=100

if do_training:
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook( tensors=tensors_to_log, every_n_secs=20 ) #every_n_iter=1000 )

    # Train the model
    epochs=5

    if False:
        mnist_classifier.fit(
          x=train_data,
          y=train_labels,
          batch_size=batch_size,
          steps=train_labels.shape[0]/batch_size * epochs,
          monitors=[logging_hook]
        )

    mnist_classifier.fit(
        input_fn=lambda: mnist_batch_input_fn(mnist.train, batch_size=batch_size, seed=42, num_epochs=epochs), 
        #steps=train_labels.shape[0] / batch_size * epochs,
        #monitors=[logging_hook],
    )

if False: # This should log 'hi[1]' to the console (not to the Jupyter window...)
    # http://stackoverflow.com/questions/37898478
    #   /is-there-a-way-to-get-tensorflow-tf-print-output-to-appear-in-jupyter-notebook-o
    a = tf.constant(1.0)
    a = tf.Print(a, [a], 'hi')
    sess = tf.Session()
    a.eval(session=sess)

# Configure the accuracy metric for evaluation
cnn_metrics = {
  "accuracy":
      learn.MetricSpec(
          metric_fn=tf.metrics.accuracy, prediction_key="classes"),
}

# Evaluate the model and print results
#cnn_eval_results = mnist_classifier.evaluate( x=eval_data, y=eval_labels, metrics=cnn_metrics)

cnn_eval_results = mnist_classifier.evaluate(
    input_fn=lambda: mnist_batch_input_fn(mnist.test, batch_size=batch_size), 
    metrics=cnn_metrics,
    #steps=eval_labels.shape[0]/batch_size,
)

print(cnn_eval_results)

train_offset = 17

image_orig = train_data[train_offset]     # This is a flat numpy array with an image in it
label_orig = train_labels[train_offset]   # This the digit label for that image

#label_target = (label_orig+1) % 10
label_target = 3

label_orig, label_target

if False: # Works, but 'old-style'
    #class_predictions = mnist_classifier.predict( x=np.array([image_orig]), batch_size=1, as_iterable=False)
    class_predictions = mnist_classifier.predict( x=image_orig, as_iterable=False)
    class_predictions['probabilities'][0]

    #class_predictions = mnist_classifier.predict( x=image_orig, outputs=['probabilities'], as_iterable=False)
    #class_predictions

def mnist_direct_data_input_fn(features_np_dict, targets_np):
    features_dict = { k:tf.constant(v) for k,v in features_np_dict.items()}
    targets = None if targets_np is None else tf.constant(targets_np)

    return features_dict, targets

class_predictions_generator = mnist_classifier.predict( 
    input_fn=lambda: mnist_direct_data_input_fn(dict(images=np.array([image_orig])), None), 
    outputs=['probabilities'],
)

for class_predictions in class_predictions_generator:
    break # Get the first one...

class_predictions['probabilities']

## Set the graph for the Inception model as the default graph,
## so that all changes inside this with-block are done to that graph.
#with model.graph.as_default():
#    # Add a placeholder variable for the target class-number.
#    # This will be set to e.g. 300 for the 'bookcase' class.
#    pl_cls_target = tf.placeholder(dtype=tf.int32)
#
#    # Add a new loss-function. This is the cross-entropy.
#    # See Tutorial #01 for an explanation of cross-entropy.
#    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=[pl_cls_target])
#
#    # Get the gradient for the loss-function with regard to
#    # the resized input image.
#    gradient = tf.gradients(loss, resized_image)

# This is the way to do it 'OLD style', where we smuggle out the information during an EVALUATE() call
if False:
    # FIGURING-IT-OUT STEP : WORKS
    def metric_accuracy(cls_targets, predictions):
      return tf.metrics.accuracy(cls_targets, predictions)

    # FIGURING-IT-OUT STEP : WORKS
    def metric_accuracy_here(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
      if labels.dtype != predictions.dtype:
        predictions = tf.cast(predictions, labels.dtype)
      is_correct = tf.to_float(tf.equal(predictions, labels))
      return tf.metrics.mean(is_correct, weights, metrics_collections, updates_collections, name or 'accuracy')

    # FIGURING-IT-OUT STEP : WORKS
    def metric_mean_here(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
      return tf.metrics.mean(labels, weights, metrics_collections, updates_collections, name or 'gradient_mean')

    # FINALLY! :: WORKS
    def metric_concat_here(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
      return tf.contrib.metrics.streaming_concat(labels, axis=0, max_size=None, 
                                         metrics_collections=metrics_collections, 
                                         updates_collections=updates_collections, 
                                         name = name or 'gradient_concat')

    model_gradient = {
    #  "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,  prediction_key="classes"), # WORKS
    #  "accuracy": learn.MetricSpec(metric_fn=metric_accuracy,      prediction_key="classes"), # WORKS
    #  "accuracy": learn.MetricSpec(metric_fn=metric_accuracy_here, prediction_key="classes"), # WORKS
    #  "accuracy": learn.MetricSpec(metric_fn=metric_mean_here,     prediction_key="classes"), # WORKS
      "gradient": learn.MetricSpec(metric_fn=metric_concat_here,   prediction_key="input_grad"), # WORKS!   
    }

    # Evaluate the model and print results  OLD-STYLE
    cnn_gradient = mnist_classifier.evaluate( 
        x=np.array([ image_orig ], dtype='float32'), y=np.array([ label_target ], dtype='int32'), 
        batch_size=1,
        #input_fn = (lambda: (np.array([ image_orig ], dtype='float32'), np.array([7], dtype='int32'))),
        metrics=model_gradient)

    #cnn_gradient = mnist_classifier.evaluate( x=image_orig, y=np.int32(7), metrics=model_gradient)

    cnn_gradient['gradient'].shape

# NEW-STYLE : We can get the data from a .PREDICT() directly (outputs=[xyz] is passed through)

def mnist_direct_data_input_fn(features_np_dict, targets_np):
    features_dict = { k:tf.constant(v) for k,v in features_np_dict.items()}
    targets = None if targets_np is None else tf.constant(targets_np)
    return features_dict, targets

tensor_prediction_generator = mnist_classifier.predict( 
    input_fn=lambda: mnist_direct_data_input_fn(
        dict(
            images=np.array([ image_orig ]),
            fake_targets=np.array([ label_target ], dtype=np.int),
        ), None), 
    outputs=['image_gradient_vs_fake_target'],
)

for tensor_predictions in tensor_prediction_generator:
    break # Get the first one...

grads = tensor_predictions['image_gradient_vs_fake_target']
grads.shape,grads.min(),grads.max()

# Plot the gradients
plt.figure(figsize=(12,3))
for i in range(1):
    plt.subplot(1, 10, i+1)
    plt.imshow(((grads+8.)/11.).reshape((28, 28)), cmap='gray', interpolation='nearest')
    plt.axis('off')

def find_adversarial_noise(image_np, cls_target, model, 
                           pixel_max=255, noise_limit=None, 
                           required_score=0.99, max_iterations=50):
    """
    Find the noise that must be added to the given image so
    that it is classified as the target-class by the given model.
    
    image_np: numpy image in correct 'picture-like' format 
    cls_target: Target class-number (integer between 0-n_classes).
    noise_limit: Limit for pixel-values in the noise (scaled for 0...255 image)
    required_score: Stop when target-class 'probabilty' reaches this.
    max_iterations: Max number of optimization iterations to perform.
    """

    # Initialize the noise to zero.
    noise = np.zeros_like( image_np )

    # Perform a number of optimization iterations to find
    # the noise that causes mis-classification of the input image.
    for i in range(max_iterations):
        print("Iteration:", i)

        # The noisy image is just the sum of the input image and noise.
        noisy_image = image_np + noise
        
        # Ensure the pixel-values of the noisy image are between
        # 0 and pixel_max like a real image. If we allowed pixel-values
        # outside this range then maybe the mis-classification would
        # be due to this 'illegal' input breaking the Inception model.
        noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=float(pixel_max))
        
        # Calculate the predicted class-scores as well as the gradient.
        #pred, grad = session.run([y_pred, gradient], feed_dict=feed_dict)
         
        tensor_prediction_generator = model.predict( 
            input_fn=lambda: mnist_direct_data_input_fn(
                dict(
                    images=np.array([ noisy_image ]),
                    fake_targets=np.array([ cls_target ], dtype=np.int),
                ), None), 
            outputs=['probabilities','logits','image_gradient_vs_fake_target'],
        )

        for tensor_predictions in tensor_prediction_generator:
            break # Get the first one...

        #tensor_predictions['image_gradient_vs_fake_target'].shape            
        
        pred   = tensor_predictions['probabilities']
        logits = tensor_predictions['logits']
        grad   = tensor_predictions['image_gradient_vs_fake_target']
        
        print( ','.join([ ("%.4f" % p) for p in pred ]))
        #print(pred.shape, grad.shape)
        
        # The scores (probabilities) for the source and target classes.
        # score_source = pred[cls_source]
        score_target = pred[cls_target]

        # The gradient now tells us how much we need to change the
        # noisy input image in order to move the predicted class
        # closer to the desired target-class.

        # Calculate the max of the absolute gradient values.
        # This is used to calculate the step-size.
        grad_absmax = np.abs(grad).max()
        
        # If the gradient is very small then use a lower limit,
        # because we will use it as a divisor.
        if grad_absmax < 1e-10:
            grad_absmax = 1e-10

        # Calculate the step-size for updating the image-noise.
        # This ensures that at least one pixel colour is changed by 7 out of 255
        # Recall that pixel colours can have 255 different values.
        # This step-size was found to give fast convergence.
        step_size = 7/255.0*pixel_max / grad_absmax

        # Print the score etc. for the source-class.
        #msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        #print(msg.format(score_source, cls_source, name_source))

        # Print the score etc. for the target-class.
        print("Target class (%d) score: %7.4f" % (cls_target, score_target, ))

        # Print statistics for the gradient.
        msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.6f}"
        print(msg.format(grad.min(), grad.max(), step_size))
        
        # Newline.
        print()
        
        # If the score for the target-class is not high enough.
        if score_target < required_score:
            # Update the image-noise by subtracting the gradient
            # scaled by the step-size.
            noise -= step_size * grad

            # Ensure the noise is within the desired range.
            # This avoids distorting the image too much.
            if noise_limit is not None:
                noise = np.clip(a     =  noise, 
                                a_min = -noise_limit/255.0*pixel_max, 
                                a_max =  noise_limit/255.0*pixel_max)
            
        else:
            # Abort the optimization because the score is high enough.
            break

    return (
        noisy_image, noise, score_target, logits
        #name_source, name_target, \
        #score_source, score_source_org, score_target
    )

np.min(image_orig), np.max(image_orig)

print(label_orig, label_target)

image_orig_sq = np.reshape(image_orig, (28,28,1))
res = find_adversarial_noise(image_orig_sq, label_target, mnist_classifier, 
                         pixel_max=1.0,   # for 0.0 ... 1.0 images (MNIST)
                         #pixel_max=255.0, # for 0..255 images (ImageNet)
                         #noise_limit=7.0,  
                         required_score=0.99, max_iterations=50)
adversarial_image, adversarial_noise, adversarial_score, adversarial_logits = res

# Plot the image, alterted image and noise
plt.figure(figsize=(12,3))
for i,im in enumerate( [image_orig, adversarial_image, adversarial_noise] ):
    plt.subplot(1, 10, 1+i)
    plt.imshow(im.reshape((28, 28)), cmap='gray', interpolation='nearest')
    plt.axis('off')

# tf.getDefaultGraph().finalize()

# Evaluate the model and gather the results.  NB: no seed, since we want to preserve the ordering

# Predictions take ~ 60secs

predictions = mnist_classifier.predict( 
    input_fn=lambda: mnist_batch_input_fn(mnist.train, batch_size=batch_size),
    outputs=['logits'],
    as_iterable=True)

train_data_logits = np.array([ p['logits'] for p in predictions ])


predictions = mnist_classifier.predict( 
    input_fn=lambda: mnist_batch_input_fn(mnist.test, batch_size=batch_size),
    outputs=['logits'],
    as_iterable=True)
eval_data_logits  = np.array([ p['logits'] for p in predictions ])

train_data_logits.shape, eval_data_logits.shape

# Optionally save the logits for quicker iteration...
logits_filename = './mnist_model/logits.pkl'

if not tf.gfile.Exists(logits_filename):
    logits_saver = ( train_data_logits, train_labels, eval_data_logits, eval_labels )
    pickle.dump(logits_saver, open(logits_filename,'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# Load the logits 
if True:
    res = pickle.load( open(logits_filename, 'rb'), encoding='iso-8859-1')
    train_data_logits, train_labels, eval_data_logits, eval_labels = res     

# Show an example #s, target_classes, and logits
print("            %s" % ( ', '.join(["%7s" % l for l in range(10)]),) )
for train_data_example in [99, 98, 84]: # all have a true label of '6'
    print("#%4d : '%d'  [ %s ]" % (
                    train_data_example,
                    train_labels[train_data_example], 
                     ', '.join(["%+7.3f" % l for l in train_data_logits[train_data_example,:]]),
         ))

# Ok, so how about the reconstruction error for the training logits that it gets wrong?

# Create an indicator function that is 1 iff the label doesn't match the best logit answer
train_labels_predicted = np.argmax( train_data_logits, axis=1 )
print("train_labels_predicted.shape     :", train_labels_predicted.shape)
print( 'predicted : ',train_labels_predicted[80:100], '\nactual    : ', train_labels[80:100] )

#train_error_indices = np.where( train_labels_predicted == train_labels, 0, 1)
train_error_indices = train_labels_predicted != train_labels
print( "Total # of bad training examples : ", np.sum( train_error_indices ) )  # [80:90]

# Gather the 'badly trained logits'
train_error_logits = train_data_logits[train_error_indices]
print("train_error_logits.shape         :", train_error_logits.shape)

train_valid_indices = train_labels_predicted == train_labels
train_valid_logits  = train_data_logits[train_valid_indices]

# Histogram various pre-processings of the input logits

#def n(x): return x
#def n(x): return ( (x - x.mean(axis=1, keepdims=True))/x.std(axis=1, keepdims=True)  )
#def n(x): return ((x - x.min(axis=1, keepdims=True))/(x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 0.0001))
#def n(x): return np.fabs(x)

def n(x):
  len_em = len_except_max = (x.shape[1]-1)
  x_max = x.max(axis=1, keepdims=True)
  x_argmax = x.argmax(axis=1)
  mean_em  = (x.sum(axis=1, keepdims=True) - x_max) / len_em
  sumsq_em = np.sum(np.square(x - mean_em), axis=1, keepdims=True)  -  np.square(x_max - mean_em)
  std_em  = np.sqrt( sumsq_em / len_em )
  y = (x - mean_em) / std_em
  y = np.clip(y, -4.0, +4.0)
  y[np.arange(x.shape[0]), x_argmax]=5.0
  return y

count, bins, patches = plt.hist(n(train_valid_logits).flatten(), 50, normed=1, facecolor='green', alpha=1.0)
count, bins, patches = plt.hist(n(train_error_logits).flatten(), 50, normed=1, facecolor='blue', alpha=0.5)

plt.xlabel('logit')
plt.ylabel('density')
#plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#plt.axis([-4, 6, 0, 0.8])
plt.grid(True)

plt.show()

def autoencoder_model_fn(features, unused_labels, mode):
  logits_dim = 10
  #hidden_dim = logits_dim
  hidden_dim = int(logits_dim*.75)

  input_layer = features['logits']  

  # One-hot on the input logit that's > 4.5
  one_hot = tf.div( tf.add( tf.sign( tf.subtract(input_layer, 4.5) ), 1.0), 2.0)
  one_hot = tf.Print(one_hot, [one_hot], message="one_hot: ", first_n=1, summarize=30 )

  # This summary is the inputs with the 'top-1' set to zero
  input_remainder = tf.subtract( input_layer, tf.multiply(one_hot, 5.0) )

  input_summary = tf.layers.dense(inputs=input_layer, units=int(logits_dim*.5), activation=tf.nn.relu)
    
  combined = tf.concat( [input_summary, one_hot], 1)
    
  # Encoder Dense Layer
    
  #dense1 = tf.layers.dense(inputs=input_layer, units=hidden_dim, activation=tf.nn.relu)
  #dense1 = tf.layers.dense(inputs=input_layer, units=logits_dim, activation=tf.nn.relu)
  #dense = tf.layers.dense(inputs=input_layer, units=hidden_dim, activation=tf.nn.elu)  # ELU!

  #dense1 = tf.layers.dense(inputs=input_layer, units=hidden_dim, activation=tf.nn.tanh)
  #dense1 = tf.layers.dense(inputs=input_layer, units=logits_dim, activation=tf.nn.tanh)
  #dense1 = tf.layers.dense(inputs=combined, units=logits_dim, activation=tf.nn.tanh)

  #dense2 = tf.layers.dense(inputs=dense1, units=hidden_dim, activation=tf.nn.tanh)
  #dense2 = tf.layers.dense(inputs=dense1, units=logits_dim*2, activation=tf.nn.tanh)
  #dense2 = tf.layers.dense(inputs=dense1, units=logits_dim, activation=tf.nn.tanh)

  #dense2 = dense1
  dense2 = combined
    
  # Add dropout operation; 0.6 probability that element will be kept
  #dropout = tf.layers.dropout(
  #    inputs=dense2, rate=0.9, training=mode == learn.ModeKeys.TRAIN)

  # Decoder Dense Layer

  #output_layer = tf.layers.dense(inputs=dropout, units=logits_dim)
  output_layer = tf.layers.dense(inputs=dense2, units=logits_dim)  # Linear activation

  loss = None
  train_op = None

  ## Calculate Loss (for both TRAIN and EVAL modes)
  #if mode != learn.ModeKeys.INFER:
  #  loss = tf.losses.mean_squared_error( input_layer, output_layer )

  if False:
      loss = tf.losses.mean_squared_error( input_layer, output_layer )

  if True:
      weighted_diff = tf.multiply( tf.subtract(1.0, one_hot), tf.subtract(input_layer, output_layer) )
      #weighted_diff = tf.multiply( 1.0, tf.subtract(input_layer, output_layer) )
      loss = tf.reduce_mean( tf.multiply (weighted_diff, weighted_diff) )

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="Adam")

  # Generate Predictions
  predictions = {
      "mse": loss,
      "regenerated":output_layer, 
      "gradient": tf.gradients(loss, input_layer),
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

mnist_autoencoder = learn.Estimator(
      model_fn=autoencoder_model_fn, model_dir="mnist_model/autoencoder")

def mnist_logit_batch_input_fn(logits, batch_size=100, seed=None, num_epochs=1):  
    # If seed is defined, this will shuffle data into batches

    all_logits  = tf.constant( logits, shape=logits.shape, verify_shape=True )
    fake_labels = tf.constant( np.zeros((logits.shape[0],)) )
    
    print("mnist_logit_batch_input_fn sizing : ", all_logits.shape, )
    
    # And create a 'feeder' to batch up the data appropriately...
    logit, label = tf.train.slice_input_producer( [ all_logits, fake_labels ], 
                                           num_epochs=num_epochs,
                                           shuffle=(seed is not None), seed=seed,
                                         )
    
    dataset_dict = dict( logits=logit, labels=label ) # This becomes pluralized into batches by .batch()
    
    batch_dict = tf.train.batch( dataset_dict, batch_size,
                                num_threads=1, capacity=batch_size*2, 
                                enqueue_many=False, shapes=None, dynamic_pad=False, 
                                allow_smaller_final_batch=False, 
                                shared_name=None, name=None)

    batch_labels = batch_dict.pop('labels')
    #batch_labels = batch_dict.pop('logits')
    
    # Return : 
    # 1) a mapping of feature columns to Tensors with the corresponding feature data, and 
    # 2) fake_labels (all 0)
    return batch_dict, batch_labels

autoenc_batch_size, autoenc_epochs = 100, 20

# Fit the autoencoder to the logits

mnist_autoencoder.fit(
    input_fn=lambda: mnist_logit_batch_input_fn( n(train_valid_logits), #train_data_logits, 
                                                batch_size=autoenc_batch_size, 
                                                seed=42, 
                                                num_epochs=autoenc_epochs), 
)

# Configure the accuracy metric for evaluation
def metric_mean_here(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
  return tf.metrics.mean(labels, weights, metrics_collections, updates_collections, name or 'gradient_mean')

autoenc_metrics = {
  "loss":learn.MetricSpec(metric_fn=metric_mean_here, prediction_key="mse"),
}

# Evaluate the model and print results
#autoencoder_eval_results = mnist_autoencoder.evaluate( x=eval_data_logits, y=eval_data_logits, metrics=auto_metrics)
autoencoder_train_results = mnist_autoencoder.evaluate( 
    input_fn=lambda: mnist_logit_batch_input_fn(n(train_valid_logits), # train_data_logits, 
                                                batch_size=train_valid_logits.shape[0], 
                                               ), 
    metrics=autoenc_metrics)

print(autoencoder_train_results)

autoencoder_eval_results = mnist_autoencoder.evaluate( 
    input_fn=lambda: mnist_logit_batch_input_fn(n(eval_data_logits), 
                                                batch_size=eval_data_logits.shape[0], 
                                               ), 
    metrics=autoenc_metrics)

print(autoencoder_eval_results)

if False:  # Double up train_error_logits to check whether mean() is working
    train_error_logits = np.vstack( [train_error_logits,train_error_logits] )
    train_error_logits.shape

# What is the mean reconstruction error for the incorrectly trained digits?

autoencoder_error_results = mnist_autoencoder.evaluate( 
    input_fn=lambda: mnist_logit_batch_input_fn(n(train_error_logits), 
                                                batch_size=train_error_logits.shape[0], 
                                               ), 
    metrics=autoenc_metrics)

print(autoencoder_error_results)

adversarial_logits

autoencoder_adversarial_results = mnist_autoencoder.evaluate( 
    input_fn=lambda: mnist_logit_batch_input_fn(n(np.array([
                    #train_data_logits[84],
                    adversarial_logits,
                ])),   
                                                batch_size=1, 
                                               ), 
    metrics=autoenc_metrics)

print(autoencoder_adversarial_results)

get_ipython().magic('pinfo tf.reduce_sum')





