# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:29:56 2014

@author: nedvedj
"""

import cv2
import matplotlib.pyplot as plt

import skimage
import skimage.io
import skimage.color
import skimage.morphology
 
from skimage.filter import threshold_otsu

import numpy as np
 
import scipy
import scipy.ndimage
plt.gray()
image = cv2.imread('../zdo2014-training3/C4C/C4c_id18621_ff184-FL_1_131030_00003365.jpg')
#image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
im = image[:]
plt.figure(1)
plt.title('Original image')
plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')

plt.figure(2)
image = image>150
plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')


image = skimage.transform.resize(im, [10, 10])

plt.figure(3)
plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')


from skimage import data
from skimage import filter

camera = im[:]
val = filter.threshold_otsu(camera)
image = camera < val

plt.figure(4)

plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')



#
#from skimage import feature
#
#from skimage.feature import corner_harris, corner_subpix, corner_peaks
#from skimage.transform import warp, AffineTransform
#
#
#tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7,
#                        translation=(210, 50))
#image = warp(image, tform.inverse, output_shape=(350, 350))
#
#coords = corner_peaks(corner_harris(image), min_distance=5)
#coords_subpix = corner_subpix(image, coords, window_size=13)
#
#plt.figure(5)
#
#plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
#
#from skimage import data, io, filter

plt.figure(6)
edges = filter.sobel(im)
plt.imshow(edges)