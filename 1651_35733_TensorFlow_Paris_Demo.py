import h2o
h2o.init(port=54321, nthreads=-1)

from h2o.estimators.deepwater import H2ODeepWaterEstimator
if not H2ODeepWaterEstimator.available(): exit

import sys, os
import os.path
import pandas as pd
import numpy as np
import random

get_ipython().magic('matplotlib inline')
from IPython.display import Image, display, HTML
import matplotlib.pyplot as plt

H2O_PATH=os.path.expanduser("~/h2o-3/")

frame = h2o.import_file(H2O_PATH + "/bigdata/laptop/deepwater/imagenet/cat_dog_mouse.csv")
print(frame.dim)
print(frame.head(5))

model = H2ODeepWaterEstimator(epochs      = 500, 
                              network     = "lenet", 
                              image_shape = [28,28],  ## provide image size
                              channels    = 3,
                              backend     = "tensorflow",
                              model_id    = "deepwater_tf_simple")

model.train(x = [0], # file path e.g. xxx/xxx/xxx.jpg
            y = 1, # label cat/dog/mouse
            training_frame = frame)

model.show()

def simple_model(w, h, channels, classes):
    import json
    import tensorflow as tf    
    # always create a new graph inside ipython or
    # the default one will be used and can lead to
    # unexpected behavior
    graph = tf.Graph() 
    with graph.as_default():
        size = w * h * channels
        x = tf.placeholder(tf.float32, [None, size])
        W = tf.Variable(tf.zeros([size, classes]))
        b = tf.Variable(tf.zeros([classes]))
        y = tf.matmul(x, W) + b

        # labels
        y_ = tf.placeholder(tf.float32, [None, classes])
     
        # accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1),                                                                                                                                                                                                                                   
                                       tf.argmax(y_, 1))                       
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # train
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        
        tf.add_to_collection("train", train_step)
        # this is required by the h2o tensorflow backend
        global_step = tf.Variable(0, name="global_step", trainable=False)
        
        init = tf.initialize_all_variables()
        tf.add_to_collection("init", init)
        tf.add_to_collection("logits", y)
        saver = tf.train.Saver()
        meta = json.dumps({
                "inputs": {"batch_image_input": x.name, "categorical_labels": y_.name}, 
                "outputs": {"categorical_logits": y.name}, 
                "metrics": {"accuracy": accuracy.name, "total_loss": cross_entropy.name},
                "parameters": {"global_step": global_step.name},
        })
        print(meta)
        tf.add_to_collection("meta", meta)
        filename = "/tmp/lenet_tensorflow.meta"
        tf.train.export_meta_graph(filename, saver_def=saver.as_saver_def())
    return filename

filename = simple_model(28, 28, 3, classes=3)

model = H2ODeepWaterEstimator(epochs                  = 500, 
                              network_definition_file = filename,  ## specify the model
                              image_shape             = [28,28],  ## provide expected image size
                              channels                = 3,
                              backend                 = "tensorflow",
                              model_id                = "deepwater_tf_custom")

model.train(x = [0], # file path e.g. xxx/xxx/xxx.jpg
            y = 1, # label cat/dog/mouse
            training_frame = frame)

model.show()

