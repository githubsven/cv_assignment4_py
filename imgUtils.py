#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:20:47 2018

Contains utilities to work with images

@author: Breixo, Sven
"""

import numpy as np
import cv2
import os

# Global variables
dataTypes = ["BrushingTeeth", "CuttingInKitchen", "JumpingJack", "Lunges", "WallPushups"]


def createData(dataDir = "ucf-101", outputDir = "training", flip = False, multiple = 1):
    """
    Creates images to feed to the Convolutional Neural Network
    It takes the frame at the middle of the video, and converts this to a png
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataFolder = dir_path + "/data/" + dataDir + "/"
    outputFolder = dir_path + "/data/" + outputDir + "/"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for type in dataTypes:
        if not os.path.exists(outputFolder + type):
            os.makedirs(outputFolder + type)

        videos = []
        for (dirpath, dirnames, filenames) in os.walk(dataFolder + type + "/"):
            videos.extend(filenames)
            break # Only needs to be executed once, because filenames is an array

        for video in videos:
            cap = cv2.VideoCapture(dataFolder + type + "/" + video)
            for i in range(multiple):
                frameNr = cap.get(7) // (multiple + 1) * i
                cap.set(1, frameNr)
                ret, frame = cap.read()
                outputFrame = resizeImage(imageToSquare(frame))
                fileName = str.split(video, ".")[0]
                cv2.imwrite(outputFolder + type + "/" + fileName + "_" + i + ".png", outputFrame)
                if flip:
                    flippedFrame = cv2.flip(outputFrame,1)
                    cv2.imwrite(outputFolder + type + "/" + fileName + "_" + i + "_flip.png", flippedFrame)


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

