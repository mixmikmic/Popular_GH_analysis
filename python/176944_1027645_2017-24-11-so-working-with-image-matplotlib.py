import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

img = mpimg.imread('Beauty-Black-White-Wallpaper.jpg')

img.shape

imgplot = plt.imshow(img)

lum_img = img[:,:,0]
imgplot = plt.imshow(lum_img)

imgplot = plt.imshow(lum_img)
imgplot.set_cmap('hot')

imgplot = plt.imshow(lum_img)
imgplot.set_cmap('spectral')

imgplot = plt.imshow(lum_img)
imgplot.set_cmap('spectral')
plt.colorbar()
plt.show()

imgplot = plt.imshow(lum_img)
imgplot.set_clim(0.0,0.7)



