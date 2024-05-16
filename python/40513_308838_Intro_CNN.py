# Load all needed libraries
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import time
import cv2
import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def plot_image(image, image2=None):
    # Show one image
    plt.subplot(121)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    else:
        plt.imshow(image, cmap = plt.get_cmap('gray'))
    plt.axis("off")
    plt.xticks([]), plt.yticks([])
    if image2 is not None:
        # Show two images
        plt.subplot(122)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            plt.imshow(image2)
        else:
            plt.imshow(image2, cmap = plt.get_cmap('gray'))
        plt.axis("off") 
        plt.xticks([]), plt.yticks([])
    plt.show()

im = cv2.imread("Lenna.png")
plot_image(im)

## Sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")
# Laplacian kernel used to detect edges
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")
# Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")
# Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")

dst = cv2.filter2D(im,-1,sharpen)
plot_image(im, dst)

dst = cv2.filter2D(im,-1,laplacian)
plot_image(im, dst)

dst = cv2.filter2D(im,-1,sobelX)
plot_image(im, dst)

dst = cv2.filter2D(im,-1,sobelY)
plot_image(im, dst)

mnist = fetch_mldata('MNIST original')
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist.data.shape[0])
X = mnist.data[p]
Y = mnist.target[p]

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X[i].reshape((28,28)), cmap='Greys_r')
    plt.axis('off')
plt.show()

X = X.astype(np.float32)/255
X_train = X[:60000].reshape((-1, 1, 28, 28))
X_test = X[60000:].reshape((-1, 1, 28, 28))
Y_train = Y[:60000]
Y_test = Y[60000:]

batch_size = 100
train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)
test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size)

# Network symbolic representation
data = mx.symbol.Variable('data')
input_y = mx.sym.Variable('softmax_label')  # placeholder for output

conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))

conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2)) 

flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500) 
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10) 
lenet = mx.symbol.SoftmaxOutput(data=fc2, label=input_y, name="softmax")

# Lenet visualization
mx.viz.plot_network(lenet)

model = mx.model.FeedForward(
    ctx = mx.cpu(),      # Run on CPU (can also use GPU: ctx = mx.gpu(0))
    symbol = lenet,       # Use the network we just defined
    num_epoch = 10,       # Train for 10 epochs
    learning_rate = 0.1,  # Learning rate
    optimizer = 'sgd',    # The optimization method is Stochastic Gradient Descent
    momentum = 0.9,       # Momentum for SGD with momentum
    wd = 0.00001)         # Weight decay for regularization

tic = time.time()
model.fit(
    X = train_iter,  # Training data set
    eval_data = test_iter,  # Testing data set. MXNet computes scores on test set every epoch
    eval_metric = ['accuracy'],  # Metric for evaluation: accuracy. Other metrics can be defined
    batch_end_callback = mx.callback.Speedometer(batch_size, 200))  # Logging module to print out progress
print("Finished training in %.0f seconds" % (time.time() - tic))

