def backprop_dense(activation, kernel, bias, relevance):
    W_p = tf.maximum(0., kernel)
    b_p = tf.maximum(0., bias)
    z_p = tf.matmul(activation, W_p) + b_p
    s_p = relevance / z_p
    c_p = tf.matmul(s_p, tf.transpose(W_p))

    W_n = tf.maximum(0., kernel)
    b_n = tf.maximum(0., bias)
    z_n = tf.matmul(activation, W_n) + b_n
    s_n = relevance / z_n
    c_n = tf.matmul(s_n, tf.transpose(W_n))

    return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)

def backprop_conv(self, activation, kernel, bias, relevance, strides, padding='SAME'):
    W_p = tf.maximum(0., kernel)
    b_p = tf.maximum(0., bias)
    z_p = nn_ops.conv2d(activation, W_p, strides, padding) + b_p
    s_p = relevance / z_p
    c_p = nn_ops.conv2d_backprop_input(tf.shape(activation), W_p, s_p, strides, padding)

    W_n = tf.minimum(0., kernel)
    b_n = tf.minimum(0., bias)
    z_n = nn_ops.conv2d(activation, W_n, strides, padding) + b_n
    s_n = relevance / z_n
    c_n = nn_ops.conv2d_backprop_input(tf.shape(activation), W_n, s_n, strides, padding)

    return activation * (self.alpha * c_p + (1 - self.alpha) * c_n)

# Bach et al.'s redistribution rule
def backprop_pool(self, activation, relevance, ksize, strides, pooling_type, padding='SAME'):
    z = nn_ops.max_pool(activation, ksize, strides, padding) + 1e-10
    s = relevance / z
    c = gen_nn_ops._max_pool_grad(activation, z, s, ksize, strides, padding)
    return activation * c


# Montavon et al.'s redistribution rule
def backprop_pool(self, activation, relevance, ksize, strides, pooling_type, padding='SAME'):
    z = nn_ops.avg_pool(activation, ksize, strides, padding) + 1e-10
    s = relevance / z
    c = gen_nn_ops._avg_pool_grad(tf.shape(activation), s, ksize, strides, padding)
    return activation * c



