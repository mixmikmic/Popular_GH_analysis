import numpy as np
import time
from evolution_strategy import *
from function import *
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('Iris.csv')
df.head()

X = PCA(n_components=2).fit_transform(MinMaxScaler().fit_transform(df.iloc[:, 1:-1]))
Y = LabelEncoder().fit_transform(df.iloc[:, -1])
one_hot = np.zeros((Y.shape[0], 3))
for i in range(Y.shape[0]):
    one_hot[i, Y[i]] = 1.0
    
train_X, test_X, train_Y, test_Y, train_label, test_label = train_test_split(X,one_hot,Y, test_size = 0.2)

X.shape

size_population = 50
sigma = 0.1
learning_rate = 0.001
epoch = 500

'''
class Deep_Evolution_Strategy:
    
    def __init__(self, weights, inputs, solutions, reward_function, population_size, sigma, learning_rate):
    
weights = array of weights, no safe checking
inputs = our input matrix
solutions = our Y matrix
reward_function = cost function, can check function.py

Check example below on how to initialize the model and train any dataset

len(activations) == len(weights)

def train(self, epoch = 100, print_every = 5, activation_function = None):
'''

weights = [np.random.randn(X.shape[1]),
           np.random.randn(X.shape[1],20),
           np.random.randn(20,one_hot.shape[1])]
activations = [sigmoid, sigmoid, softmax]
deep_evolution = Deep_Evolution_Strategy(weights, train_X, train_Y, cross_entropy, size_population, sigma, learning_rate)
deep_evolution.train(epoch=1000,print_every = 50, activation_function = activations)

predicted= np.argmax(deep_evolution.predict(deep_evolution.get_weight(), train_X, activation_function = activations),axis=1)
print(metrics.classification_report(predicted, np.argmax(train_Y, axis=1), target_names = ['flower 1', 'flower 2', 'flower 3']))

predicted= np.argmax(deep_evolution.predict(deep_evolution.get_weight(), test_X, activation_function = activations),axis=1)
print(metrics.classification_report(predicted, np.argmax(test_Y, axis=1), target_names = ['flower 1', 'flower 2', 'flower 3']))

accuracy_test = np.mean(predicted == np.argmax(test_Y, axis=1))


plt.figure(figsize=(15,10))
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.argmax(deep_evolution.predict(deep_evolution.get_weight(), np.c_[xx.ravel(), yy.ravel()], activation_function = activations),axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
plt.title('decision boundary, accuracy validation: %f'%(accuracy_test))
plt.show()



