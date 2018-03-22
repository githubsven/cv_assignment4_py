import numpy as np
import cv2
import tensorflow as tf
import os
import imgUtils

#Edit paper: https://www.sharelatex.com/5152532135sdnhgphhqsbd

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # The frames are 90x90 pixels, and have one grayscale color channel
  pass


def processVideos(dataDir = "training"):
    """
    Read videos and transform them to input data
    Returns images converted to np.arrays in the images dictionary ordered per label
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataFolder = dir_path + "/data/" + dataDir + "/"
    dataTypes = imgUtils.dataTypes

    images = []
    labels = []
    for type in dataTypes:
        files = []
        for (dirpath, dirnames, filenames) in os.walk(dataFolder + type + "/"):
            files.extend(filenames)
            break  # Only needs to be executed once, because filenames is an array

        for file in files:
            image = cv2.imread(dataFolder + type + "/" + file, 0) # Load an color image in grayscale
            images.append(image)
            labels.append(type)

    boundary = 0.8 * len(images) # The seperation between training data and evaluation data is at 80%
    train_data = images[0:boundary]
    train_labels = labels[0:boundary]
    eval_data = images[boundary:]
    eval_labels = images[boundary:]

    return train_data, train_labels, eval_data, eval_labels


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
    train_data, train_labels, eval_data, eval_labels = processVideos()
    train()
    test()


if __name__ == "__main__":
    #imgUtils.createData()
    #imgUtils.createData("own")
    tf.app.run()