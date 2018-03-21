import numpy as np
import cv2
import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

#Edit paper: https://www.sharelatex.com/5152532135sdnhgphhqsbd

def processVideos():
    """
    TODO Read videos and transform them to input data
    """
    pass # Does nothing

def train():
    """
    TODO Train the NN
    """
    pass # Does nothing

def test():
    """
    TODO Test the NN
    """
    pass # Does nothing


def main():
    processVideos()
    train()
    test()


def createData():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataFolder = dir_path + "/data/ucf-101/"
    outputFolder = dir_path + "/data/training/"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    dataTypes = ["BrushingTeeth", "CuttingInKitchen", "JumpingJack", "Lunges", "WallPushups"]

    for type in dataTypes:
        if not os.path.exists(outputFolder + type):
            os.makedirs(outputFolder + type)
        videos = []
        for (dirpath, dirnames, filenames) in os.walk(dataFolder + type + "/"):
            videos.extend(filenames)
            break

        for video in videos:
            cap = cv2.VideoCapture(dataFolder + type + "/" + video)
            halfpoint_frame = round(cap.get(7) / 2)
            cap.set(1, halfpoint_frame)
            ret, frame = cap.read()
            fileName = str.split(video, ".")[0]
            cv2.imwrite(outputFolder + type + "/" + fileName + ".png", frame)
            break


if __name__ == "__main__":
    createData()
    #tf.app.run()