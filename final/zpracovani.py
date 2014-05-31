# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:29:56 2014

@author: nedvedj
"""
plt.close('all')
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
pole = np.array([])

plt.gray()
image = cv2.imread('../zdo2014-training3/C4C/C4c_id18621_ff184-FL_1_131030_00003365.jpg')
#image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
im = image[:]
plt.figure(1)
plt.title('Original image')
plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')

#%% Prevod RGB2GRAY pro detekci hran a dalsi zpracovani
cernobily = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#%% Změna velikosti
import skimage.transform

image = skimage.transform.resize(image, [50, 50])

plt.figure(2)
plt.title('Resize')
plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')

#%% Vyriznuti objektu kolem stredu obrazu
image = image[int(image.shape[0]/2)-20:int(image.shape[0]/2)+20, int(image.shape[1]/2)-20:int(image.shape[1]/2)+20]
plt.figure(3)
plt.title('Vyriznuti')
plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
pole = np.append(pole, image.reshape(-1))

#%% Detekce hran

cernobily = cv2.Canny(cernobily,60,60)
plt.figure(4)
plt.title('Detekce hran')
plt.imshow(cernobily)

kernel_big = skimage.morphology.diamond(1) 
cernobily = skimage.morphology.binary_dilation(cernobily, kernel_big) # Na detekovane hrany pouzijeme dilataci
cernobily = cv2.GaussianBlur(cernobily,(5,5), 5) # Gausovska filtrace pro odstraneni nezadoucich objektu

plt.figure(5)
plt.title('Dilatace a vyhlazeni')
plt.imshow(cernobily, cmap=plt.cm.gray, interpolation='nearest')

#%% Vyriznuti objektu kolem stredu obrazu
cernobily = cernobily[int(cernobily.shape[0]/2)-20:int(cernobily.shape[0]/2)+20, int(cernobily.shape[1]/2)-20:int(cernobily.shape[1]/2)+20]
plt.figure(6)
plt.title('Vyriznuti')
plt.imshow(cernobily, cmap=plt.cm.gray, interpolation='nearest')
pole = np.append(pole, cernobily.reshape(-1))

#%% Prace s filtrem
from skimage import filter

camera = image[:]
val = filter.threshold_otsu(camera)
image = camera < val

plt.figure(7)
plt.title('Pouziti OTSU filtru')
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

edges = filter.sobel(camera)
plt.figure(8)
plt.title('Hledání hran přes sobel')
plt.imshow(edges)