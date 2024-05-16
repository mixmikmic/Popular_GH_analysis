import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

bitfinex_prices = pd.read_csv("./bitfinex-USDprice.csv")

bitfinex_prices.head()

bitfinex_prices.tail()

bitcoin = pd.read_csv('./googletrends-bitcoin.csv')

bitcoin.head()

weekly_price = []

for week in bitcoin['Week']:
    w = bitfinex_prices.loc[bitfinex_prices['Date'] == week]
    price = w['High'].values
    if len(price) == 0:
        weekly_price.append(np.nan)
    else:
        weekly_price.append(price[0])
     

bitcoin['Price'] = weekly_price

bitcoin

# Drop weeks for which there are missing values 
bitcoin = bitcoin.dropna(axis=0, how='any')

data = bitcoin[['bitcoin','Price']].values

plt.plot(data[:,1],data[:,0], "bo")
plt.xlabel('price (USD)')
plt.ylabel('Google Trends Score')
plt.show()

# Normalize the data
X = (data[:,1] - np.mean(data[:,1]))/np.std(data[:, 1])
Y = (data[:,0] - np.mean(data[:,0]))/np.std(data[:, 0])

plt.plot(X,Y, "bo")
plt.xlabel('price (USD)')
plt.ylabel('Google Trends Score')
plt.show()

def simplest_neural_net(x, y, epochs, learning_rate):
    weights = np.array([0, 0])
    bias = 1.
    for e in range(epochs):
        gradient = np.array([0., 0.])
        for i in range(len(x)):
            xi = x[i]
            xi = np.array([bias, xi])
            yi = y[i]
            
            h = np.dot(weights, xi) - yi
            gradient += 2*xi*h
        
        weights = weights - learning_rate*gradient
    return weights[1], weights[0]

# Here the ideal values for slope and y-intercept are converged upon
m, b = simplest_neural_net(X,Y,100, 1e-3)
target_m = m
target_b = b

x_points = np.linspace(np.min(X), np.max(X), 10)
line = b + m*x_points

plt.plot(X,Y, 'bo', x_points, line, 'r-')
plt.show()

from sklearn.metrics import mean_squared_error

def evaluation_fxn(x, y, learn_rate, ideal_m, ideal_b):
    x_points = np.linspace(np.min(x), np.max(x), 10)
    ideal_line = ideal_m*x_points + ideal_b
    m, b = simplest_neural_net(x,y,5,learn_rate)
    test_line = m*x_points + b
    
    return 1 - mean_squared_error(ideal_line, test_line)
    

# Make some inital guesses about the learning rate and evaluate them
# The Gaussian Process will be fit to this data initially.
guesses = [6e-3,1e-3,1e-4]

outcomes = [evaluation_fxn(X,Y,guess, target_m, target_b)
                     for guess in guesses]

from sklearn.gaussian_process import GaussianProcess
import warnings
warnings.filterwarnings('ignore')

plt.plot(guesses,outcomes,'ro')
plt.xlabel('learning rate guesses')
plt.ylabel('score out of 1')
plt.show()

def hyperparam_selection(guesses, outcomes):
    guesses = np.array(guesses)
    outcomes = np.array(outcomes)
    gp = GaussianProcess(corr='squared_exponential',
                         theta0=1e-1, thetaL=1e-3, thetaU=1)
    
    gp.fit(guesses.reshape((-1,1)), outcomes)
    
    x = np.linspace(np.min(guesses), np.max(guesses), 10)
    
    mean, var = gp.predict(x.reshape((-1,1)), eval_MSE=True)
    std = np.sqrt(var)
    
    expected_improv_lower = mean - 1.96 * std
    expected_improv_upper = mean + 1.96 * std
    
    acquisition_curve = expected_improv_upper - expected_improv_lower
    
    
    idx = acquisition_curve.argmax()
    
    next_param = x[idx]
    
    plt.plot(guesses,outcomes,'ro', label='observations')
    plt.plot(x,mean, 'b--', label='posterior mean')
    plt.plot(x, expected_improv_lower, 'g--', label='variance')
    plt.plot(x, expected_improv_upper, 'g--')
    plt.plot(x, acquisition_curve, 'k--', label='acquisition fxn')
    plt.plot(x[idx],acquisition_curve[idx], 'yX', label='next guess')
    plt.xlabel('learning rate')
    plt.ylabel('score out of 1')
    plt.legend(loc='best')
    plt.show()
    
    return next_param

for _ in range(10):
    
    try:
        new_learning_rate = hyperparam_selection(guesses,outcomes) 
    except:
        print("optimal learning rate found")
        break
    
    guesses.append(new_learning_rate)
    score = evaluation_fxn(X,Y,new_learning_rate, target_m, target_b)
    print("Suggested learning rate: ",new_learning_rate)
    outcomes.append(score)

