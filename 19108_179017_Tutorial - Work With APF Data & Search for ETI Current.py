get_ipython().magic('matplotlib inline')

import numpy as np
import pyfits as pf
import matplotlib
from matplotlib import pyplot as plt
import lmfit
from lmfit import minimize, Parameters, report_fit, fit_report
from IPython.display import Image

apf_file = pf.open('ucb-amp194.fits copy')

header = apf_file[0].header
print repr(header)

print "Right Ascension: " + header['RA']
print "Declination: " + header['DEC']
print "Target Object: " + header['TOBJECT']

image = apf_file[0].data

cookies = np.array([[1,2,3],[4,5,6]])
print cookies

print cookies[0,0]

plt.imshow(image)
plt.title('2D array data')

plt.imshow(image, vmin = np.median(image), vmax = np.median(image) * 1.2, origin = 'lower')
plt.title('2D array data w/ contrast')

image_rot = np.rot90(image)
plt.figure(figsize=(20,20))
plt.imshow(image_rot, vmin = np.median(image_rot), vmax = np.median(image_rot) * 1.2, origin = 'lower')
plt.title('2D array w/ contrast and rotated')

image_flip = np.fliplr(image_rot)
plt.figure(figsize=(20,20))
plt.imshow(image_flip, vmin = np.median(image_flip), vmax = np.median(image_flip) * 1.2, origin = 'lower')
plt.title('2D array w/ contrast and rotated AND flipped')

plt.figure(figsize=(20,20))
plt.imshow(image_flip, cmap = 'gray', 
           vmin = np.median(image_flip), vmax = np.median(image_flip) * 1.2, origin = 'lower')
plt.title('Final 2D array (Our Spectrum!)')

patch = image_flip[1683:1688, 2200:2800]
# ^ Cutout of our 2D array, like a patch out of a quilt
plt.imshow(patch, cmap = 'gray', 
           vmin = np.median(image_flip), vmax = np.median(image_flip) *1.2, 
           origin = 'lower')
plt.title('small patch [1683:1688, 2200:2800] of telluric lines')

plt.imshow(image_flip[1683:1688, 2200:2800], cmap = 'gray', aspect = 'auto', 
           vmin = np.median(image_flip), vmax = np.median(image_flip) *1.2, origin = 'lower')
plt.title('small patch [1683:1688, 2200:2800] of telluric lines')

plt.imshow(image_flip[1683:1688, 2200:2800], cmap = 'gray', aspect = 'auto', 
           interpolation = 'nearest', vmin = np.median(image_flip), 
           vmax = np.median(image_flip) *1.2, origin = 'lower')
plt.title('small patch [1683:1688, 2200:2800] of telluric lines')

patch = image_flip[1683:1688, 2200:2800]
patch.size

telluric_1D = np.sum(patch, axis = 0)

plt.plot(telluric_1D)

plt.figure(figsize =(10,10))
plt.subplot(2,1,1)
plt.imshow(image_flip[1683:1688,2200:2800], cmap = 'gray', aspect = 'auto', 
           interpolation = 'nearest', origin = 'lower',
           vmin = np.median(image_flip), vmax = np.median(image_flip) *1.2)
plt.subplot(2,1,2)
plt.plot(telluric_1D)

bias = np.median(image_flip[-30:])
print bias

plt.figure(figsize=(10,5))
telluric_1D_adj = telluric_1D - (5*bias)
plt.plot(telluric_1D_adj)
plt.title('Telluric Absorption (Adjusted) [1683:1688, 2200:2800]')

def cut_n_zoom(x1,x2,y1,y2):
    plt.figure(figsize=(10,10))
    plt.imshow(image_flip[x1:x2, y1:y2], cmap = 'gray', aspect = 'auto', 
               vmin = np.median(image), vmax = np.median(image) *1.2, origin = 'lower')
    plt.show()

#cutting out the patch with the absorption feature
h_alpha_patch = image_flip[1491:1506,1500:2500] 
#take the sum along the columns, and subtract 15 biases
h_alpha_patch_1D_without_bias = np.sum(h_alpha_patch, axis = 0) - bias*15 

# Plotting H-alpha absorption line
plt.figure(figsize=(10,10))
plt.plot(np.sum(h_alpha_patch, axis = 0) - bias*15)
plt.title('H-alpha')

plt.figure(figsize=(10,10))
plt.imshow(image_flip[1333:1348,1200:2200], cmap = 'gray', aspect = 'auto', 
           interpolation = 'nearest', origin = 'lower',
           vmin = np.median(image_flip), vmax = np.median(image_flip) *1.2)

Na_D_patch = image_flip[1333:1348, 1200:2200]
Na_D_patch_1D = np.sum(Na_D_patch, axis = 0) - bias*15
plt.figure(figsize=(10,10))
plt.plot(Na_D_patch_1D)
plt.title('Na-D lines')

plt.imshow(image_flip[1650:1750], aspect = 'auto', origin = 'lower', cmap = "gray", 
           interpolation = 'nearest', vmin = np.median(image_flip), vmax = np.median(image_flip) *1.1)

#Load the reduced .fits file and extracting the data
apf_reduced = pf.open('ramp.194.fits copy')
reduced_image_fits = apf_reduced[0].data
#Plot an image of the reduced data
plt.figure(figsize=(10,6))
plt.imshow(reduced_image_fits, cmap = "gray", origin = "lower", aspect = "auto", 
           vmin = np.median(reduced_image_fits), vmax = np.median(reduced_image_fits) *1.1)
plt.title("Reduced Spectrum")
#Plot an image of the raw data
plt.figure(figsize=(10,6))
plt.imshow(image_flip, cmap = "gray", origin = "lower", aspect = "auto", 
           vmin = np.median(image_flip), vmax = np.median(image_flip) *1.1)
plt.title("Full Spectrum")
print "Whereas the raw data array has dimensions %s pixels by %s pixels," % image_flip.shape
print "this reduced data array has dimensions %s pixels by %s pixels." % reduced_image_fits.shape

print "Right Ascension: " + header['RA']
print "Declination: " + header['DEC']
print "Target Object: " + header['TOBJECT']

header_reduced = apf_reduced[0].header

print "Reduced - Right Ascension: " + header_reduced['RA']
print "Reduced - Declination: " + header_reduced['DEC']
print "Reduced - Target Object: " + header_reduced['TOBJECT']

text = open('order_coefficients copy.txt', "r")
lines = text.read().splitlines()
print lines[0]

a0 = float(lines[0][6:13].strip())
a1 = float(lines[0][17:26].strip())
a2 = float(lines[0][27:39].strip())
a3 = float(lines[0][40:52].strip())
a4 = float(lines[0][54:].strip())
print a0, a1, a2, a3, a4 

coeff_array = np.zeros((79,5))

for i in range(len(lines)):
    a0 = float(lines[i][6:13].strip())
    a1 = float(lines[i][17:26].strip())
    a2 = float(lines[i][27:39].strip())
    a3 = float(lines[i][40:52].strip())
    a4 = float(lines[i][54:].strip())
    coeffs_one_line = np.array([a0,a1,a2,a3,a4])
    coeff_array[i] += coeffs_one_line

#Plots raw image
plt.figure(figsize=(12,8))
plt.imshow(image_flip, cmap = "gray", origin = "lower", 
        aspect = "auto", vmin = np.median(image_flip), 
        vmax = np.median(image_flip) *1.1)
#Sets array of x values, which the polynomials can then be plotted with
x = np.arange(0,4608)
#Plots each polynomial function over the raw image
for i in range(coeff_array[:,0].size):
    a0 = coeff_array[i,0]
    a1 = coeff_array[i,1]
    a2 = coeff_array[i,2]
    a3 = coeff_array[i,3]
    a4 = coeff_array[i,4]
    #Plots each order of coefficients to fit a fourth-degree polynomial
    plt.plot(x, a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)
    #Sets the limit on the x-axis and the y-axis shown in the plots
    plt.xlim(0,4608)
    plt.ylim(0,2080)
plt.title("Raw image with polynomial functions overplotted")

#Array of increasing x values
x = np.arange(0, 4608).astype(float)
#Empty array to fill in with y values from polynomials
y_values = np.zeros((79,4608))
#Empty array to fill in to create our reduced spectrum
poly_reduced_image = np.zeros((79,4608))
#Iteration loop that adds y values to the y_values array and 
#adds pixel values to the reduced_image array
for i in range(coeff_array[:,0].size):
    a0 = coeff_array[i,0]
    a1 = coeff_array[i,1]
    a2 = coeff_array[i,2]
    a3 = coeff_array[i,3]
    a4 = coeff_array[i,4]
    for j in range(x.size):
        y = a0 + a1*x[j] + a2*x[j]**2 + a3*x[j]**3 + a4*x[j]**4
        y_values[i,j] = y
        y = int(round(y))
        #We sum the pixel with three pixels above and three pixels below to ensure that 
        #we're including all of the important pixels in our reduced image
        poly_reduced_image[i,j] = int(np.sum(image_flip[y-3:y+4,j], 
            axis = 0)-7*bias)
plt.figure(figsize=(10,7))
plt.imshow(poly_reduced_image, cmap = "gray", origin = "lower", 
    aspect = "auto", vmin = np.median(poly_reduced_image), 
    vmax = np.median(poly_reduced_image) *1.1)

plt.figure(figsize=(12,8))
plt.subplot(2, 1, 1)
plt.imshow(poly_reduced_image, cmap = "gray", origin = "lower", 
    aspect = "auto", vmin = np.median(poly_reduced_image), 
    vmax = np.median(poly_reduced_image) *1.1)
plt.title("Reduced Image through Polyfit Technique")
plt.subplot(2, 1, 2)
plt.title("Reduced Image File")
plt.imshow(reduced_image_fits, cmap = "gray", origin = "lower", 
    aspect = "auto", vmin = np.median(reduced_image_fits), 
    vmax = np.median(reduced_image_fits) *1.1)

print poly_reduced_image[53,2000]
print reduced_image_fits[53,2000]

plt.figure(figsize=(12,8))
plt.subplot(2, 1, 1)
plt.plot(poly_reduced_image[53])
plt.title("Reduced Image (Polyfit) H-alpha")

plt.subplot(2, 1, 2)
plt.plot(reduced_image_fits[53])
plt.title("Reduced Image File H-alpha")

plt.figure(figsize=(12,8))
plt.subplot(2, 1, 1)
plt.plot(poly_reduced_image[53])
plt.ylim(0,1200)
plt.title("Reduced Image (Polyfit) H-alpha")

plt.subplot(2, 1, 2)
plt.plot(reduced_image_fits[53])
plt.ylim(0,1200)
plt.title("Reduced Image File H-alpha")

from lmfit.models import GaussianModel

wave = pf.open('apf_wave.fits copy')
wave_values = wave[0].data

x = wave_values[53,0:4000]
y = poly_reduced_image[53,0:4000]

plt.figure(figsize=(10,6))
plt.plot(x,y)

wave_h_alpha = wave_values[53,1942-500:1942+500]

reduced_h_alpha = poly_reduced_image[53,1942-500:1942+500]

plt.plot(wave_h_alpha,reduced_h_alpha)

left_median = np.median(reduced_h_alpha[0:50])
right_median = np.median(reduced_h_alpha[-50:])
median = (right_median + left_median)/2
print median

reduced_h_alpha_shifted = (reduced_h_alpha / median) - 1

x = wave_h_alpha
y = reduced_h_alpha_shifted

mod = GaussianModel()
pars = mod.guess(y,x=x)
out = mod.fit(y, pars, x=x)
plt.figure(figsize=(10,10))
plt.plot(x, y)
plt.plot(x, out.best_fit)
print out.fit_report()
print 'Center at ' + str(out.best_values['center']) + ' Angstroms for our created reduced image.'

reduced_image_provided_h_alpha = reduced_image_fits[53,1942-500:1942+500]
left_median = np.median(reduced_image_provided_h_alpha[0:50])
right_median = np.median(reduced_image_provided_h_alpha[-50:])
median = (right_median + left_median)/2
print median

reduced_provided_h_alpha_shifted = (reduced_image_provided_h_alpha / median) - 1
x = wave_h_alpha
y = reduced_provided_h_alpha_shifted

mod = GaussianModel()
pars = mod.guess(y, x=x)
out = mod.fit(y, pars, x=x)
plt.figure(figsize=(10,10))
plt.plot(x, y)
plt.plot(x, out.best_fit)
print out.fit_report(min_correl = 0.25)
print 'Center at ' + str(out.best_values['center']) + ' Angstroms for the reduced image we were provided.'

def cosmic_ray_spot(patch_1D):
    plt.figure(figsize=(10,10))
    plt.plot(patch_1D, color = 'b')
    for i in range(5, patch_1D.size - 5):
        if ((patch_1D[i]>patch_1D[i-1]) and (patch_1D[i]>patch_1D[i+1]) and (patch_1D[i]>(bias*1.25))
            and (patch_1D[i-3]<=(bias*1.25)) and (patch_1D[i+3]<=(bias*1.25))):
            half_max = ((patch_1D[i]) + (patch_1D[i+5] + patch_1D[i-5])/2)/2
            left_side = np.where(patch_1D[:i] <= half_max)
            left_mark = left_side[0][-1]
            right_side = np.where(patch_1D[i:] <= half_max)
            right_mark = right_side[0][0] + i
            peak_x = right_mark - ((right_mark - left_mark)/2)
            plt.axvline(x=peak_x, ymin = np.min(patch_1D) - 1000, color = 'r', linestyle = '-.')

cosmic_ray_spot(poly_reduced_image[53])

cosmic_ray_spot(Na_D_patch_1D)



