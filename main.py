import numpy as np
import cv2
import tensorflow as tf
import os
import imgUtils

#Edit paper: https://www.sharelatex.com/5152532135sdnhgphhqsbd

tf.logging.set_verbosity(tf.logging.INFO)

def processVideos():
    """
    TODO Read videos and transform them to input data
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataFolder = dir_path + "/data/training/"
    dataTypes = ["BrushingTeeth", "CuttingInKitchen", "JumpingJack", "Lunges", "WallPushups"]

    images = {}
    for type in dataTypes:
        files = []
        for (dirpath, dirnames, filenames) in os.walk(dataFolder + type + "/"):
            files.extend(filenames)
            break  # Only needs to be executed once, because filenames is an array

        images[type] = []
        for file in files:
            image = cv2.imread(dataFolder + type + "/" + file, 0) # Load an color image in grayscale
            images[type].append(image)

    return images


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


def main(argv):
    print(processVideos())
    train()
    test()


if __name__ == "__main__":
    #imgUtils.createData()
    tf.app.run()