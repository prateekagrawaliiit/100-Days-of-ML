# -*- coding: utf-8 -*-
# @Author: prateek
# @Date:   2020-09-03 22:36:59
# @Last Modified by:   prateek
# @Last Modified time: 2020-09-03 22:37:01


import numpy as np
import imageio as io
import scipy.ndimage
import cv2

image = "doggo.png"

def grayscale(rgb):
    return np.dot(rgb[...,:3],[0.299, 0.587, 0.114])

def dodge(front, back):
    result = front*255/(255-back)
    result[result>255]=255
    result[back==255]=255
    return result.astype('uint8')

# Declaring some variables

img = io.imread(image)
gs = grayscale(img)
g=255-gs

b = scipy.ndimage.filters.gaussian_filter(g,sigma=10)
d=dodge(b,gs)

print("Sketch Successfull")

cv2.imwrite('sketch.png', d)