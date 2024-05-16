import numpy as np
import re
import sys

from scipy import misc

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

DIR = "/home/barsana/datasets/kitti/odometry-dataset/sequences/05/precomputed-depth-dispnet-png/"

for i in range(1, 100):
    fpath = "{}{:06d}.png".format(DIR, i)
    fpath_pretty = "{}{:06d}-viridis.png".format(DIR, i)
    img = plt.imread(fpath)
    
    plt.imsave(fpath_pretty, img, cmap='viridis')
    


D_ELAS = "/home/barsana/datasets/kitti/odometry-dataset/sequences/06/image_2/"

elas = D_ELAS + "000028_disp.pgm"
elas_dispmap = plt.imread(elas)
plt.imshow(elas_dispmap, cmap='viridis')
plt.imsave(D_ELAS + "0000028_disp-viridis.png", elas_dispmap, cmap='viridis')



