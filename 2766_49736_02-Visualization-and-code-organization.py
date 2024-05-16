from __future__ import absolute_import, division, print_function



# uncomment the bottom line in this cell, change the final line of 
# the loaded script to `mpld3.display()` (instead of show).

# %load http://mpld3.github.io/_downloads/linked_brush.py

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import mpld3
from mpld3 import plugins, utils

data = load_iris()
X = data.data
y = data.target

# dither the data for clearer plotting
X += 0.1 * np.random.random(X.shape)

fig, ax = plt.subplots(4, 4, sharex="col", sharey="row", figsize=(8, 8))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                    hspace=0.1, wspace=0.1)

for i in range(4):
    for j in range(4):
        points = ax[3 - i, j].scatter(X[:, j], X[:, i],
                                      c=y, s=40, alpha=0.6)

# remove tick labels
for axi in ax.flat:
    for axis in [axi.xaxis, axi.yaxis]:
        axis.set_major_formatter(plt.NullFormatter())

# Here we connect the linked brush plugin
plugins.connect(fig, plugins.LinkedBrush(points))

# mpld3.show()
mpld3.display()

















get_ipython().magic('matplotlib inline')

import mpld3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context('poster')
# sns.set_style('whitegrid') 
sns.set_style('darkgrid') 
plt.rcParams['figure.figsize'] = 12, 8  # plotsize 

def sinplot(flip=1, ax=None):
    """Demo plot from seaborn."""
    x = np.linspace(0, 14, 500)
    for i in range(1, 7):
        ax.plot(x, np.sin(-1.60 + x + i * .5) * (7 - i) * flip, label=str(i))

mpld3.enable_notebook()

fig, ax = plt.subplots(figsize=(12, 8))
sinplot(ax=ax)
ax.set_ylabel("y-label")
ax.set_xlabel("x-label")
fig.tight_layout()

mpld3.disable_notebook()



