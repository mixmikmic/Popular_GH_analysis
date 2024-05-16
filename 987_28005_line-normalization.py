get_ipython().magic('pylab inline')
from pylab import imshow
from scipy.ndimage import filters,interpolation
import ocrolib
from ocrolib import lineest

#Configure the size of the inline figures
figsize(8,8)

image = 1-ocrolib.read_image_gray("../tests/010030.bin.png")
image = interpolation.affine_transform(image,array([[0.5,0.015],[-0.015,0.5]]),offset=(-30,0),output_shape=(200,1400),order=0)

imshow(image,cmap=cm.gray)
print image.shape

#reload(lineest)
mv = ocrolib.lineest.CenterNormalizer()
mv.measure(image)

print mv.r
plot(mv.center)
plot(mv.center+mv.r)
plot(mv.center-mv.r)
imshow(image,cmap=cm.gray)

dewarped = mv.dewarp(image)

print dewarped.shape
imshow(dewarped,cmap=cm.gray)

imshow(dewarped[:,:320],cmap=cm.gray,interpolation='nearest')

normalized = mv.normalize(image,order=0)

print normalized.shape
imshow(normalized,cmap=cm.gray)



