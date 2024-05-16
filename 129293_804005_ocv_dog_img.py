#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import OpenCV and other needed Python modules
import numpy as np
import cv2

# Load the source image
img = cv2.imread('Intel_Wall.jpg')
# Create a named window to show the source image
cv2.namedWindow('Source Image', cv2.WINDOW_NORMAL)
# Display the source image
cv2.imshow('Source Image',img)

# Load the logo image
dog = cv2.imread('Intel_Logo.png')
# Create a named window to handle intermediate outputs and resizing
cv2.namedWindow('Result Image', cv2.WINDOW_NORMAL)

# To put logo on top-left corner, create a Region of Interest (ROI)
rows,cols,channels = dog.shape
roi = img[0:rows, 0:cols ]
# Print out the dimensions of the logo...
print(dog.shape)

# Convert the logo to grayscale
dog_gray = cv2.cvtColor(dog,cv2.COLOR_BGR2GRAY)
# The code below in this cell is only to display the intermediate result and not in the script
from matplotlib import pyplot as plt
plt.imshow(dog_gray)
plt.show()

# Create a mask of the logo and its inverse mask
ret, mask = cv2.threshold(dog_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# The code below in this cell is only to display the intermediate result and not in the script
plt.imshow(mask_inv)
plt.show()

# Now blackout the area of logo in ROI
img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Now just extract the logo
dog_fg = cv2.bitwise_and(dog,dog,mask = mask)
# The code below in this cell is only to display the intermediate result and not in the script
plt.imshow(dog_fg)
plt.show()

# Next add the logo to the source image
dst = cv2.add(img_bg,dog_fg)
img[0:rows, 0:cols ] = dst

# Display the Result
cv2.imshow('Result Image',img)
# Wait until windows are dismissed
cv2.waitKey(0)

# Release all resources used
cv2.destroyAllWindows()

