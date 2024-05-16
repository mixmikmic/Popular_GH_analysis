import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
import random
style.use('fivethirtyeight')

#plotting some random points 
x = np.array([1,2,3,4,5,6],dtype=np.float64)  #single independent variable
y = np.array([6,4,5,2,1,1],dtype=np.float64)  #dependent variable

plt.scatter(x,y)
plt.show()

def best_fit_slope_intercept(x,y):
    m = (mean(x)*mean(y)-mean(x*y))/(mean(x)**2 - mean(x*x))
    b = mean(y) - m*mean(x)
    return m,b

#find the best fit line for the given point set
m,b = best_fit_slope_intercept(x,y) 

#finding the regression line
reg_line = [(m*i)+b for i in x]

#plotting the given points
plt.scatter(x,y)

#plotting the regression line
plt.plot(x,reg_line)
plt.show()

def squared_error(y_orig,y_line):
    return sum((y_orig-y_line)**2)

def coeff_det(y_orig,y_line):
    y_mean_line = [mean(y_orig) for i in y_orig]
    sse = squared_error(y_orig,y_line)
    sst = squared_error(y_orig,y_mean_line)
    return 1-(sse/sst)

print(coeff_det(y,reg_line))

# hm :  size of dataset (number of points)
# var:  variance in the y-value 
# step: change in y-value for consecutive x-values
# correlation: whether y-value increases with x-value or not.

def create_dataset(hm,var,step=2,correlation=False):
    val =1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-var,var)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]        
    return np.array(xs,dtype=np.float64), np.array(ys,dtype=np.float64)

#Creating dataset
xs,ys = create_dataset(100,20,2,'pos')

#Fitting the best fit line
m,b = best_fit_slope_intercept(xs,ys)
reg_line = [(m*i)+b for i in xs]

#Plotting the points and the regression line
plt.scatter(xs,ys)
plt.plot(xs,reg_line)
plt.show()

#Finding the coefficient of determination
print(coeff_det(ys,reg_line))

