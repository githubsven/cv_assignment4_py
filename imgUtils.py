#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:20:47 2018

@author: breixo
"""
import numpy as np

def cropImage(img):
    nRows, nCols, nChannels = np.shape(img)
    minimum = min(nRows, nCols)

    centerRows = nRows//2 # Integer division
    centerCols = nCols//2
    margin = minimum//2
    croppedImg = img[centerRows-margin:centerRows+margin, \
                     centerCols-margin:centerCols+margin]
    return croppedImg