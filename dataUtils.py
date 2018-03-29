from collections import deque
import numpy as np

# Divides the data evenly in parts.
# data: List of data points
# randomize: If true the data is randomized before dividing
# n_parts: Number of parts in which divide. If the number of parts is bigger
#    than the number of data items the result will be divided in less parts
# Used for 10-fold cross-validation.
def divide_data(data, n_parts = 10):
    length = len(data)
    stops = []
    for i in range(1, n_parts):
        stops.append(length * i / n_parts)
    stops = deque(stops)

    division = []
    current = []
    for i in range(length):
        if len(stops) > 0 and i >= stops[0]:
            division.append(current)
            current = []
            stops.popleft()
        current.append(data[i])
    division.append(current)
    return division


def prepare_training_test_data(divided_data, i_test):
    test_data = divided_data[i_test]
    training_data = []
    for i in range(len(divided_data)):
        if i != i_test:
            training_data += divided_data[i]
    return np.asarray(training_data), np.asarray(test_data)


def simple_split(images, labels, boundary = 0.8):
    bound = round(boundary * len(images))  # The seperation between training data and evaluation data is at 80%
    train_data = np.asarray(images[0:bound], dtype=np.float16)
    train_labels = np.asarray(labels[0:bound])
    eval_data = np.asarray(images[bound:], dtype=np.float16)
    eval_labels = np.asarray(labels[bound:])

    return train_data, train_labels, eval_data, eval_labels