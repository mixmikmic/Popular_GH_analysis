import tensorflow as tf

W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

sess.run(linear_model, {x: [1, 2, 3, 4]})

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
sess.run(loss, {x: x_train, y: y_train})

fixW = tf.assign(W, [-1])
fixb = tf.assign(b, [1])
sess.run([fixW, fixb])
sess.run(loss, {x: x_train, y: y_train})

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

sess.run([W, b, loss], {x: x_train, y: y_train})

logs_path = '/home/ubuntu/tensorflow-logs'
# make the directory if it does not exist
get_ipython().system('mkdir -p $logs_path')

tf.reset_default_graph()

# Model parameters
W = tf.Variable([.3], tf.float32, name='W')
b = tf.Variable([-.3], tf.float32, name='b')
# Model input and output
x = tf.placeholder(tf.float32, name='x')
linear_model = W * x + b
y = tf.placeholder(tf.float32, name='y')
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run(
    [W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

summary_writer = tf.summary.FileWriter(
    logs_path, graph=sess.graph)
summary_writer.close()

import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1, 2, 3, 4])
y = np.array([0, -1, -2, -3])
input_fn = tf.contrib.learn.io.numpy_input_fn(
    {'x': x}, y, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)
estimator.evaluate(input_fn=input_fn)

def model(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", shape=[1], dtype=tf.float64)
    b = tf.get_variable("b", shape=[1], dtype=tf.float64)
    y = W * features['x'] + b
    
    loss = tf.reduce_sum(tf.square(y - labels))
    
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
                     tf.assign_add(global_step, 1))
    return tf.contrib.learn.ModelFnOps(
        mode=mode, predictions=y, loss=loss, train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn(
    {'x': x}, y, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
estimator.evaluate(input_fn=input_fn, steps=10)



