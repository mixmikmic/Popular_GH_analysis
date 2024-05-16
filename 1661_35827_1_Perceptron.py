import random
import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

def hardlim(a):
    if a > 0.0:
        return 1.
    else:
        return 0.

class Perceptron(object):
    '''
    simple feed forward perceptron with a hard limit activation
    trained with the perceptron learning rule
    '''
    def __init__(self):
        self.alpha = None
        self.w = None

    def response(self, X):
        '''
        perceptron response
        :param X: input vector
        :return: perceptron out
        '''
        a = hardlim(np.dot(self.w.T, X))
        return a

    def updateWeight(self, X, error):
        '''
        update the vector of input weights
        :param X: input data
        :param error: prediction != true
        :return: updated weight vector
        '''
        self.w += self.alpha * error * X

    def train(self, X, y, alpha, iterations):
        '''
        trains perceptron on vector data by looping each row and updating the weight vector
        :param X: input data
        :param y: correct y value
        :return: updated parameters
        '''
        # initialize the learning rate and count data size
        self.alpha = alpha
        num_examples, num_features = np.shape(X)

        # set up bias
        bias = np.ones(shape=(num_examples,1))
        X = np.hstack((X, bias))

        # initialize weight vector
        self.w = np.random.rand(num_features + 1)
        
        error_count = []
        for i in range(iterations):
            for j in range(num_examples):
                prediction = self.response(X[j])
                error = int(y[j]) - prediction 
                self.updateWeight(X[j], error)
                error_count.append(error)
                
        error_count = np.array(error_count)
        plt.plot(error_count)
        plt.ylim([-1,1])
        plt.show()

# generate training data
X = np.array([[-1,-1],
              [-1,1],
              [1,-1],
              [1,1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

# plot the points
plt.plot(X[0][0],X[0][1], 'ro', markersize=15)
plt.plot(X[1][0],X[1][1], 'ro', markersize=15)
plt.plot(X[2][0],X[2][1], 'ro', markersize=15)
plt.plot(X[3][0],X[3][1], 'bo', markersize=15)
plt.axis([-1.1,1.1,-1.1,1.1])
plt.show()

model = Perceptron()
model.train(X, y, alpha=0.1, iterations=10)    

n = np.linalg.norm(model.w[0:2])
ww = (model.w[0:2]) / n
ww1 = [float(ww[1])* 2., -1.0*float(ww[0])* 2.] 
ww2 = [-1.0*float(ww[1]) * 2., float(ww[0]) * 2.]
plt.plot([ww1[0]-model.w[2],ww2[0]-model.w[2]], [ww1[1]-model.w[2],ww2[1]-model.w[2]])
plt.plot(X[0][0],X[0][1], 'ro', markersize=15)
plt.plot(X[1][0],X[1][1], 'ro', markersize=15)
plt.plot(X[2][0],X[2][1], 'ro', markersize=15)
plt.plot(X[3][0],X[3][1], 'bo', markersize=15)
plt.axis([-1.1,1.1,-1.1,1.1])
plt.show()





