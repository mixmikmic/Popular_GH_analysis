import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# just checking if tensorflow is using GPU device
with tf.device('/device:GPU:0'):
    x = tf.constant([1.0, 2.0, 3.0], shape = [1,3], name = 'x')
    y = tf.constant([4.0, 5.0, 6.0], shape = [3,1], name = 'y')
    z = tf.matmul(x,y)
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
    print(sess.run(z))

