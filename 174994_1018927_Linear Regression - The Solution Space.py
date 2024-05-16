import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets as skd
from IPython.display import display
from sympy import MatrixSymbol, Matrix
from numpy.linalg import inv
from ipywidgets import interact, interactive, fixed, interact_manual
from mpl_toolkits.mplot3d import Axes3D

sp.init_printing(order='rev-lex',use_latex='mathjax')

def mse(yPredict,yActual):
    return np.square(yPredict.T-yActual.T).mean()
    
def h(slope,y_intercept):
    return lambda x:slope*x+y_intercept

def interactiveLine(slope,y_intercept,noise=50):
    X,Y = skd.make_regression(100,1,random_state=0, noise=noise)
    
    ys = np.apply_along_axis(h(slope,y_intercept), 0, X)
    
    plt.figure(figsize=(10,10))
    
    plt.title("Mean Squared Error: {0}".format(mse(ys,Y)),fontsize=15)
    plt.ylabel("Dependent Variable (Y)",fontsize=15)
    plt.xlabel("Independent Variable (X)",fontsize=15)
    plt.scatter(X,Y)
    plt.plot(X,ys)

    plt.show()
    

interact(interactiveLine, slope=(-150,150),y_intercept=(-100,100),noise=(0,150));

X,Y = skd.make_regression(100,1,random_state=0, noise=50)

plt.scatter(X,Y)
plt.show()

# This can be done much more efficiently with matrix algebra, but this is the clearest way of doing this. 
# Enumerate over the entire space of linear functions within the ranges. 
# This cell will take a while to compute. 

# Uses h(slope,yInter)
def enumFnsAndMSE(possibleSlopes,possibleYInter,X_Data,Y_Data,scaleFactor=1):
    errorOverLines = []

    for slope in possibleSlopes:
        row = []

        for yInter in possibleYInter:
            lFN = h(slope,yInter)
            regressionYs = np.apply_along_axis(lFN, 0, X_Data)
            row.append( mse(regressionYs,Y_Data)/scaleFactor )

        errorOverLines.append(row)
        
    return errorOverLines

possibleSlopes = range(-150,150)
possibleYInter = range(-150,150)

errorOverLines = enumFnsAndMSE(possibleSlopes,possibleYInter,X,Y,600)

# Plot figure 
xx, yy = np.mgrid[-150:150, -150:150]

fig = plt.figure(figsize=(14,14))
ax = fig.gca(projection='3d',facecolor='gray')

ax.set_xlabel('Slope',fontSize=16)
ax.set_ylabel('Y-Intercept',fontSize=16)
ax.set_zlabel('Mean Squared Error',fontSize=16)

ax.plot_surface(xx, yy, errorOverLines, cmap=plt.cm.Reds, linewidth=0.2)
plt.show()

def findMinimum(eol):
    rowLen = len(eol)
    colLen = len(eol[0])
    
    minVal = 100000000
    minCord = []
    
    for r_i in range(rowLen):
        for c_i in range(colLen):
            if(eol[r_i][c_i] < minVal):
                minVal = eol[r_i][c_i] 
                # We split the range in half to account for the negative values. 
                minCord = [r_i-rowLen/2,c_i-colLen/2]
                
    return [minCord,minVal]
            

bestLine = findMinimum(errorOverLines)
minSlope,minIntercept = findMinimum(errorOverLines)[0]
ys = np.apply_along_axis(h(minSlope,minIntercept), 0, X)

plt.scatter(X,Y)
plt.plot(X,ys)

plt.show()
print("MSE of line: {0}".format(bestLine[1]))

