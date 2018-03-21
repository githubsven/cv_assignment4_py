import numpy as np
import cv2
import tensorflow as tf
import imgUtils

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


def main(argv):
    processVideos()
    train()
    test()


if __name__ == "__main__":
    imgUtils.createData()
    #tf.app.run()