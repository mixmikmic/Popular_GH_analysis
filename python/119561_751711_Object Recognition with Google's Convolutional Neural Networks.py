import tensorflow as tf
from datasets import dataset_utils
import os

# Base url
TF_MODELS_URL = "http://download.tensorflow.org/models/"

# Modify this path for a different CNN
INCEPTION_V3_URL = TF_MODELS_URL + "inception_v3_2016_08_28.tar.gz"
INCEPTION_V4_URL = TF_MODELS_URL + "inception_v4_2016_09_09.tar.gz"

# Directory to save model checkpoints
MODELS_DIR = "models/cnn"

INCEPTION_V3_CKPT_PATH = MODELS_DIR + "/inception_v3.ckpt"
INCEPTION_V4_CKPT_PATH = MODELS_DIR + "/inception_v4.ckpt"

# Make the model directory if it does not exist
if not tf.gfile.Exists(MODELS_DIR):
    tf.gfile.MakeDirs(MODELS_DIR)
 
# Download the appropriate model if haven't already done so
if not os.path.exists(INCEPTION_V3_CKPT_PATH):    
    dataset_utils.download_and_uncompress_tarball(INCEPTION_V3_URL, MODELS_DIR)
    
if not os.path.exists(INCEPTION_V4_CKPT_PATH):
    dataset_utils.download_and_uncompress_tarball(INCEPTION_V4_URL, MODELS_DIR)

from preprocessing import inception_preprocessing
# This can be modified depending on the model used and the training image dataset

def process_image(image):
    root_dir = "images/"
    filename = root_dir + image
    with open(filename, "rb") as f:
        image_str = f.read()
        
    if image.endswith('jpg'):
        raw_image = tf.image.decode_jpeg(image_str, channels=3)
    elif image.endswith('png'):
        raw_image = tf.image.decode_png(image_str, channels=3)
    else: 
        print("Image must be either jpg or png")
        return 
    
    image_size = 299 # ImageNet image size, different models may be sized differently
    processed_image = inception_preprocessing.preprocess_image(raw_image, image_size,
                                                             image_size, is_training=False)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        raw_image, processed_image = sess.run([raw_image, processed_image])
        
    return raw_image, processed_image.reshape(-1, 299, 299, 3)

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

def plot_color_image(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image.astype(np.uint8), interpolation='nearest')
    plt.axis('off')

raw_bison, processed_bison = process_image('bison.jpg')

plot_color_image(raw_bison)

raw_sombrero, processed_sombrero = process_image('sombrero.jpg')

plot_color_image(raw_sombrero)

print(raw_bison.shape, processed_bison.shape)

from datasets import imagenet
from tensorflow.contrib import slim
from nets import inception

'''
predict(image, version) bFunction takes in the name of the image and optionally the network to use for predictions
Currently, the only options for the net are Inception V3 and Inception V4.
Plots the raw image and displays the top-10 class predictions.
'''

def predict(image, version='V3'):
    tf.reset_default_graph()
    
    # Process the image 
    raw_image, processed_image = process_image(image)
    class_names = imagenet.create_readable_names_for_imagenet_labels()
    
    # Create a placeholder for the images
    X = tf.placeholder(tf.float32, [None, 299, 299, 3], name="X")
    
    '''
    inception_v3 function returns logits and end_points dictionary
    logits are output of the network before applying softmax activation
    '''
    
    if version.upper() == 'V3':
        model_ckpt_path = INCEPTION_V3_CKPT_PATH
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            # Set the number of classes and is_training parameter  
            logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=False)
            
    elif version.upper() == 'V4':
        model_ckpt_path = INCEPTION_V4_CKPT_PATH
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            # Set the number of classes and is_training parameter
            # Logits 
            logits, end_points = inception.inception_v4(X, num_classes=1001, is_training=False)
            
    
    predictions = end_points.get('Predictions', 'No key named predictions')
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, model_ckpt_path)
        prediction_values = predictions.eval({X: processed_image})
        
    try:
        # Add an index to predictions and then sort by probability
        prediction_values = [(i, prediction) for i, prediction in enumerate(prediction_values[0,:])]
        prediction_values = sorted(prediction_values, key=lambda x: x[1], reverse=True)
        
        # Plot the image
        plot_color_image(raw_image)
        plt.show()
        print("Using Inception_{} CNN\nPrediction: Probability\n".format(version))
        # Display the image and predictions 
        for i in range(10):
            predicted_class = class_names[prediction_values[i][0]]
            probability = prediction_values[i][1]
            print("{}: {:.2f}%".format(predicted_class, probability*100))
    
    # If the predictions do not come out right
    except:
        print(predictions)

predict('bison.jpg', version='V3')

predict('bison.jpg', version='V4')

predict('sombrero.jpg', version='V4')

predict('tiger-shark.jpg', version='V3')

predict('albatross.jpg', version='V4')

predict('squirrel.jpg', version='V4')

predict('basketball.jpg', version='V4')

predict('basketball_game.jpg', version='V4')

predict('giraffe.jpg', version='V4')

predict('calculator.jpg', version='V4')

predict('basketball_game.jpg', version='V4')



