import numpy as np

import tensorflow as tf

# Step 1- import data

points = np.genfromtxt('data.csv', delimiter=',')

# separate out data
x = points[:,0]
y = points[:,1]

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.scatter(x,y)
plt.show()

# randomly initialize weights and biases

slope = tf.Variable([1], dtype=tf.float32)
intercept = tf.Variable([0], dtype=tf.float32)

# define hyper parameter
rate = 0.000001
num_iterations = 1000

# define data placeholders

x_val = tf.placeholder(tf.float32)
y_val = tf.placeholder(tf.float32)

# define the model

predicted_y = slope*x_val + intercept

# find your error

error = tf.reduce_sum(tf.square(predicted_y - y_val))

# find grdient and update weights

optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)

# minimize the error
train = optimizer.minimize(error)

# initialize tf session

sess = tf.Session()

# initialize only the variables

init = tf.global_variables_initializer()

sess.run(init)

# train your model for certain number of iterations

for p in range(num_iterations):
    
    sess.run(train, feed_dict={x_val:x, y_val:y})

sess.run(slope)

sess.run(intercept)

def predict(m,b,x_vals):
    
    ys = []
    
    for i in range(len(x_vals)):
        
        predicted_y = m*x_vals[i]+b
        ys.append(predicted_y)
        
    return ys

final_predicted_slope = sess.run(slope)
final_predicted_intercept = sess.run(intercept)

final_predicted_intercept

plt.scatter(x,y)
plt.plot(x, predict(final_predicted_slope, final_predicted_intercept,x))
plt.show()

predict(final_predicted_slope, final_predicted_intercept, [65])



