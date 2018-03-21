#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:20:47 2018

Contains utilities to work with images

@author: breixo
"""

import numpy as np
import cv2

def imageToSquare(img):
    """
    Crops the borders of an image to get a square shape
    """
    nRows, nCols, nChannels = np.shape(img)
    minimum = min(nRows, nCols)

    centerRows = nRows//2 # Integer division
    centerCols = nCols//2
    margin = minimum//2
    croppedImg = img[centerRows-margin:centerRows+margin, \
                     centerCols-margin:centerCols+margin]
    return croppedImg

def resizeImage(img, newSize = (90, 90)):
    """
    OpenCV function wrapper
    """
    return cv2.resize(img, newSize)

def showImage(img, name = "Image"):
    """
    Wrapper to easily show an image in the OpenCV style. Avoids stupid crashes
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

