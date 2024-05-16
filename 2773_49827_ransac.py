get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

def iters_needed(k, p=.99, w = .5):
    """
    Returns the number of ransac iterations needed.
    :param k : Minimal set of points.
    :param w : Percentage of inliers.
    :param p : Desired Accuracy
    """
    return np.log(1-p)/np.log(1 - np.power(w, k))

x = range(1, 10)
y = [iters_needed(x_i) for x_i in x]
plt.plot(x, y)
plt.title('Number of iterations VS Minimal set of inliers')
plt.xlabel('Number of iterations')
plt.ylabel('Minimal set of inliers')
plt.show()



