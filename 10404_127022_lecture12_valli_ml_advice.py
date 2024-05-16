get_ipython().magic('matplotlib inline')
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.datasets import make_classification
X, y = make_classification(1000, n_features=5, n_informative=2, 
                           n_redundant=2, n_classes=2, random_state=0)

from pandas import DataFrame
df = DataFrame(np.hstack((X, y[:, None])), 
               columns = list(range(5)) + ["class"])

df[:5]

# Pairwise feature plot

_ = sns.pairplot(df[:50], vars=[0, 1, 2, 3, 4], hue="class", size=1.5)

# Correlation Plot
plt.figure(figsize=(10, 10));
_ = sns.heatmap(df.corr(), annot=False)

from IPython.display import Image
Image(filename='images/sklearn_sheet.png', width=800, height=600) 

from IPython.display import Image
Image(filename='images/azure_sheet.png', width=800, height=600) 

# adapted from http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_loss_functions.html
def plot_loss_functions():
    xmin, xmax = -4, 4
    xx = np.linspace(xmin, xmax, 100)
    plt.plot(xx, xx ** 2, 'm-',
             label="Quadratic loss")
    plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], 'k-',
             label="Zero-one loss")
    plt.plot(xx, 1/(1 + np.exp(xx)), 'b-',
             label="Sigmoid loss")
    plt.plot(xx, np.where(xx < 1, 1 - xx, 0), 'g-',
             label="Hinge loss")
    plt.plot(xx, np.log2(1 + np.exp(-xx)), 'r-',
             label="Log loss")
    plt.plot(xx, np.exp(-xx), 'c-',
             label="Exponential loss")
    plt.ylim((0, 8))
    plt.legend(loc="best")
    plt.xlabel(r"Decision function $f(x)$")
    plt.ylabel("$L(y, f)$")

# Demonstrate some loss functions
plot_loss_functions()

import pylab as pl
def plot_fit(x, y, p, show):
    xfit = np.linspace(0, 2, 1000)
    yfit = np.polyval(p, xfit)
    if show:
        pl.scatter(x, y, c='k')
        pl.plot(xfit, yfit)
        pl.hold('on')
        pl.xlabel('x')
        pl.ylabel('y')

def calc_errors(p):
    x = np.linspace(0, 2, 1000)
    errs = []
    for i in x:
        errs.append(abs(np.polyval(p, i) - np.sin(np.pi * i)) ** 2)
    return errs

def polyfit_sin(degree, iterations, num_points=2, show=True):
    total = 0
    l = []
    coeffs = []
    errs = [0] * len(np.linspace(0, 2, 1000))
    for i in range(iterations):
        np.random.seed()
        x = 2 * np.random.random(num_points) # Pick random points from the sinusoid with co-domain [0, 2)
        y = np.sin(np.pi * x)
        p = np.polyfit(x, y, degree)  
        y_poly = [np.polyval(p, x_i) for x_i in x]  
        plot_fit(x, y, p, show)
        total += sum(abs(y_poly - y) ** 2) # calculate Squared Error (Squared Error) 
        coeffs.append(p)
        errs = np.add(calc_errors(p), errs)
    return total / iterations, errs / iterations, np.mean(coeffs, axis = 0), coeffs

# Estimate two points of sin(pi * x) with a constant once
# Ignore return values for now, we will return to these later
_, _, _, _ = polyfit_sin(0, 1)

# Estimate two points of sin(pi * x) with a constant 5 times
_, _, _, _ = polyfit_sin(0, 5)

# Estimate two points of sin(pi * x) with a constant 100 times
_, _, _, _ = polyfit_sin(0, 100)

# Estimate two points of sin(pi * x) with a constant 500 times
MSE, errs, mean_coeffs, coeffs_list = polyfit_sin(0, 500)

x = np.linspace(0, 2, 1000)

# Polynomial with mean coeffs.
p = np.poly1d(mean_coeffs)

# Calculate Bias
errs_ = []
for i in x:
    errs_.append(abs(np.sin(np.pi * i) - np.polyval(p, i)) ** 2)
print("Bias: "  + str(np.mean(errs_)))

x = np.linspace(0, 2, 1000)

diffs = []

# Calculate Variance
for coeffs in coeffs_list:
    p = np.poly1d(coeffs)
    for i in x:
        diffs.append(abs(np.polyval(np.poly1d(mean_coeffs), i) - np.polyval(p, i)) ** 2)  
print("Variance: "  + str(np.mean(diffs)))

# Error Bars plot

xfit = np.linspace(0, 2, 1000)
yfit = np.polyval(np.poly1d(mean_coeffs), xfit)
pl.scatter(xfit, yfit, c='g')
pl.hold('on')
pl.plot(xfit, np.sin(np.pi * xfit))
pl.errorbar(xfit, yfit, yerr = errs, c='y', ls="None", zorder=0)
pl.xlabel('x')
pl.ylabel('y')

# Estimate two points of sin(pi * x) with a line 1 times
MSE, _, _, _ = polyfit_sin(1, 1)
print(MSE)
# Note: Perfect fit! (floating point math cause non-zero MSE)

# Estimate two points of sin(pi * x) with a line 5 times
_, _, _, _ = polyfit_sin(1, 5)

# Estimate two points of sin(pi * x) with a line 100 times
_, _, _, _ = polyfit_sin(1, 100)

# Estimate two points of sin(pi * x) with a line 500 times
MSE, errs, mean_coeffs, coeffs = polyfit_sin(1, 500)

x = np.linspace(0, 2, 1000)

# Polynomial with mean coeffs.
p = np.poly1d(mean_coeffs)

# Calculate Bias
errs_ = []
for i in x:
    errs_.append(abs(np.sin(np.pi * i) - np.polyval(p, i)) ** 2)
print("Bias: " + str(np.mean(errs_)))

x = np.linspace(0, 2, 1000)

diffs = []

# Calculate Variance
for coeff in coeffs:
    p = np.poly1d(coeff)
    for i in x:
        diffs.append(abs(np.polyval(np.poly1d(mean_coeffs), i) - np.polyval(p, i)) ** 2)  
print("Variance: "  + str(np.mean(diffs)))

# Error bars plot

xfit = np.linspace(0, 2, 1000)
yfit = np.polyval(np.poly1d(mean_coeffs), xfit)
pl.scatter(xfit, yfit, c='g')
pl.hold('on')
pl.plot(xfit, np.sin(np.pi * xfit))
pl.errorbar(xfit, yfit, yerr = errs, c='y', ls="None", zorder=0)
pl.xlabel('x')
pl.ylabel('y')

# Image from Andrew Ng's Stanford CS229 lecture titled "Advice for applying machine learning"
from IPython.display import Image
Image(filename='images/HighVariance.png', width=800, height=600)

# Testing error still decreasing as the training set size increases. Suggests increasing the training set size.
# Large gap Between Training and Test Error.

# Image from Andrew Ng's Stanford CS229 lecture titled "Advice for applying machine learning"
from IPython.display import Image
Image(filename='images/HighBias.png', width=800, height=600)

# Training error is unacceptably high.
# Small gap between training error and testing error.

