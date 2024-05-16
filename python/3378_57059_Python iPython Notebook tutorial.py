# General Setups and Imports

get_ipython().magic('matplotlib inline')
from matplotlib.pyplot import plot, title, xlabel, ylabel
from numpy import linspace, sin

x = linspace(0, 20, 1000)  # 100 evenly-spaced values from 0 to 50
y = sin(x)

plot(x, y)
xlabel('this is X')
ylabel('this is Y')
title('My Plot')

