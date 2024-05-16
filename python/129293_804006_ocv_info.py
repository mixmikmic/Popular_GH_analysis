#!/usr/bin/env python2

from __future__ import print_function

import numpy as np

print('Numpy Version:', np.__version__)

import cv2

print('OpenCV Version:', cv2.__version__)

import matplotlib as mpl
import os
import sys

print('Matplotlib Version:', mpl.__version__)
print(sys.version)

try:
    pyth_path = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    pyth_path = []

print('Python Environment Variable - PYTHONPATH:', pyth_path)

try:
    ocv2_path = os.environ['OPENCV_DIR']
except KeyError:
    ocv2_path = []
    
try:
    ocv2_vers = os.environ['OPENCV_VER']
except KeyError:
    ocv2_path = []

print('OpenCV Environment Variable - OPENCV_DIR:', ocv2_path)
print('OpenCV Environment Variable - OPENCV_VER:', ocv2_vers)

try:
    ffmp_path = os.environ['FFMPEG_BIN']
except KeyError:
    ffmp_path = []

print('FFMPEG Environment Variable - FFMPEG_BIN:', ffmp_path)

