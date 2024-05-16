# import input_data
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

mnist.train.num_examples

mnist.test.num_examples

mnist.train.images.shape

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().magic('pylab inline')

first_image_array = mnist.train.images[0, ]
image_length = int(np.sqrt(first_image_array.size))
first_image = np.reshape(first_image_array, (-1, image_length))

first_image.shape

plt.imshow(first_image, cmap = cm.Greys_r)
plt.show()





