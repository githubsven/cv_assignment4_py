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
import random

# Global variables
dataTypes = ["BrushingTeeth", "CuttingInKitchen", "JumpingJack", "Lunges", "WallPushups"]
random.seed(1)


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
                cv2.imwrite(outputFolder + type + "/" + fileName + "_" + str(i) + ".png", outputFrame)
                if flip:
                    flippedFrame = cv2.flip(outputFrame,1)
                    cv2.imwrite(outputFolder + type + "/" + fileName + "_" + str(i) + "_flip.png", flippedFrame)


def processVideos(dataDir = "training", shuffleData = True):
    """
    Read videos and transform them to input data
    Returns images converted to np.arrays in the images dictionary ordered per label
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataFolder = dir_path + "/data/" + dataDir + "/"

    images = []
    labels = []
    for type in range(len(dataTypes)):
        currentType = dataTypes[type]
        files = []
        for (dirpath, dirnames, filenames) in os.walk(dataFolder + currentType + "/"):
            files.extend(filenames)
            break  # Only needs to be executed once, because filenames is an array

        for file in files:
            image = cv2.imread(dataFolder + currentType + "/" + file)
            images.append(image)
            labels.append(type)

    if shuffleData:
        shuffle(images, labels)

    return images, labels

def prepareTestOurVideos(trainingDir = "our_training",
                         testingDir = "our_testing", shuffleData = True):
    
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataFolderTraining = dir_path + "/data/" + trainingDir + "/"

    imagesTraining = []
    labelsTraining = []
    for type in range(len(dataTypes)):
        currentType = dataTypes[type]
        files = []
        for (dirpath, dirnames, filenames) in os.walk(dataFolderTraining + currentType + "/"):
            files.extend(filenames)
            break  # Only needs to be executed once, because filenames is an array

        for file in files:
            image = cv2.imread(dataFolderTraining + currentType + "/" + file)
            imagesTraining.append(image)
            labelsTraining.append(type)
    
    dataFolderTesting = dir_path + "/data/" + testingDir + "/"
    
    imagesTesting = []
    labelsTesting = []
    for type in range(len(dataTypes)):
        currentType = dataTypes[type]
        files = []
        for (dirpath, dirnames, filenames) in os.walk(dataFolderTesting + currentType + "/"):
            files.extend(filenames)
            break  # Only needs to be executed once, because filenames is an array

        for file in files:
            image = cv2.imread(dataFolderTesting + currentType + "/" + file)
            imagesTesting.append(image)
            labelsTesting.append(type)

    if shuffleData:
        shuffle(imagesTraining, labelsTraining)
        shuffle(imagesTesting, labelsTesting)
    train_data = np.asarray(imagesTraining, dtype=np.float16)
    train_labels = np.asarray(labelsTraining)
    test_data = np.asarray(imagesTesting, dtype=np.float16)
    test_labels = np.asarray(labelsTesting)
    return train_data, train_labels, test_data, test_labels

def original_shuffle(images, labels):
    for i in range(len(images)):
        firstPos = random.randint(0, len(images) - 1)
        secondPos = random.randint(0, len(images) - 1)
        tempImage = images[firstPos]
        tempLabel = labels[firstPos]
        images[firstPos] = images[secondPos]
        labels[firstPos] = labels[secondPos]
        images[secondPos] = tempImage
        labels[secondPos] = tempLabel

def shuffle(images, labels):
    size = len(images)
    order = list(range(size))
    random.shuffle(order)
    copy_images = images.copy()
    copy_labels = labels.copy()
    for i in range(size):
        firstPos = i
        secondPos = order[i]
        images[firstPos] = copy_images[secondPos]
        labels[firstPos] = copy_labels[secondPos]

def imageToSquare(img):
    """
    Crops the borders of an image to get a square shape
    """
    nRows, nCols, nChannels = np.shape(img)
    minimum = min(nRows, nCols)

    centerRows = nRows // 2 # Integer division
    centerCols = nCols // 2
    margin = minimum // 2
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

