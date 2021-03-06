import tensorflow as tf
import os
import imgUtils, dataUtils
import numpy as np

#Edit paper: https://www.sharelatex.com/5152532135sdnhgphhqsbd

tf.logging.set_verbosity(tf.logging.INFO)

#Copied from https://stackoverflow.com/questions/46326376/tensorflow-confusion-matrix-in-experimenter-during-evaluation
def eval_confusion_matrix(labels, predictions):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=5)
        con_matrix_sum = tf.Variable(tf.zeros(shape=(5,5), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])

        update_op = tf.assign_add(con_matrix_sum, con_matrix)
        return tf.convert_to_tensor(con_matrix_sum), update_op


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # The frames are 90x90 pixels, and have one grayscale color channel
  input_layer = tf.reshape(features["x"], [-1, 90, 90, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 90, 90, 1]
  # Output Tensor Shape: [batch_size, 86, 86, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 86, 86, 32]
  # Output Tensor Shape: [batch_size, 43, 43, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 90, 90, 1]
  # Output Tensor Shape: [batch_size, 86, 86, 32]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 86, 86, 32]
  # Output Tensor Shape: [batch_size, 43, 43, 32]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 90, 90, 1]
  # Output Tensor Shape: [batch_size, 86, 86, 32]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 86, 86, 32]
  # Output Tensor Shape: [batch_size, 43, 43, 32]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 19, 19, 64]
  # Output Tensor Shape: [batch_size, 19 * 19 * 64]
  pool3_flat = tf.reshape(pool3, [-1, 11 * 11 * 32])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 19 * 19 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 5]
  logits = tf.layers.dense(inputs=dropout, units=5)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  error = tf.reduce_mean(loss, name="loss_tensor")

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
    "precision": tf.metrics.precision(labels=labels, predictions=predictions["classes"]),
    "confusion_matrix": eval_confusion_matrix(labels=labels, predictions=predictions["classes"]),
    "recall": tf.metrics.recall(labels=labels, predictions=predictions["classes"])
  }

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_9x9_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # The frames are 90x90 pixels, and have one grayscale color channel
  input_layer = tf.reshape(features["x"], [-1, 90, 90, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 90, 90, 1]
  # Output Tensor Shape: [batch_size, 82, 82, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[9, 9],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 82, 82, 32]
  # Output Tensor Shape: [batch_size, 41, 41, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 41, 41, 32]
  # Output Tensor Shape: [batch_size, 37, 37, 32]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 86, 86, 32]
  # Output Tensor Shape: [batch_size, 43, 43, 32]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 90, 90, 1]
  # Output Tensor Shape: [batch_size, 86, 86, 32]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 86, 86, 32]
  # Output Tensor Shape: [batch_size, 43, 43, 32]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 19, 19, 64]
  # Output Tensor Shape: [batch_size, 19 * 19 * 64]
  pool3_flat = tf.reshape(pool3, [-1, 11 * 11 * 32])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 19 * 19 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 5]
  logits = tf.layers.dense(inputs=dropout, units=5)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      # "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "error":  tf.reduce_mean(loss, name="loss_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
    "precision": tf.metrics.precision(labels=labels, predictions=predictions["classes"]),
    "confusion_matrix": eval_confusion_matrix(labels=labels, predictions=predictions["classes"]),
    "recall": tf.metrics.recall(labels=labels, predictions=predictions["classes"])
  }
  #fScore = 2 * eval_metric_ops["precision"] * eval_metric_ops["recall"] / (eval_metric_ops["precision"] + eval_metric_ops["recall"])
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_4_layer_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # The frames are 90x90 pixels, and have one grayscale color channel
  input_layer = tf.reshape(features["x"], [-1, 90, 90, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 90, 90, 1]
  # Output Tensor Shape: [batch_size, 82, 82, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 82, 82, 32]
  # Output Tensor Shape: [batch_size, 41, 41, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 41, 41, 32]
  # Output Tensor Shape: [batch_size, 37, 37, 32]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 86, 86, 32]
  # Output Tensor Shape: [batch_size, 43, 43, 32]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 90, 90, 1]
  # Output Tensor Shape: [batch_size, 86, 86, 32]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 86, 86, 32]
  # Output Tensor Shape: [batch_size, 43, 43, 32]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 19, 19, 64]
  # Output Tensor Shape: [batch_size, 19 * 19 * 64]
  pool4_flat = tf.reshape(pool4, [-1, 10 * 10 * 32])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 19 * 19 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 5]
  logits = tf.layers.dense(inputs=dropout, units=5)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      # "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "error":  tf.reduce_mean(loss, name="loss_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
    "precision": tf.metrics.precision(labels=labels, predictions=predictions["classes"]),
    "confusion_matrix": eval_confusion_matrix(labels=labels, predictions=predictions["classes"]),
    "recall": tf.metrics.recall(labels=labels, predictions=predictions["classes"])
  }
  #fScore = 2 * eval_metric_ops["precision"] * eval_metric_ops["recall"] / (eval_metric_ops["precision"] + eval_metric_ops["recall"])
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def train(classifier, data, labels):
    """
    TODO Train the NN
    """
    tensors_to_log = {"error": "loss_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=2)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data},
        y=labels,
        batch_size=64,
        num_epochs=3,
        shuffle=False)

    classifier.train(
        input_fn=train_input_fn,
        steps=200,
        hooks=[logging_hook])


def test(classifier, data, labels):
    """
    TODO Test the NN
    """
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data},
        y=labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    eval_results["F-Score"] = 2 * eval_results["precision"] * eval_results["recall"] / (eval_results["precision"] + eval_results["recall"])
#    print(eval_results)
    return eval_results

def experiment_n_fold(images, labels, nFolds, name = "", doTrain = True):
    divided_images = dataUtils.divide_data(images, nFolds)
    divided_labels = dataUtils.divide_data(labels, nFolds)
    if (name != ""):
        name = "_" + name
    
    all_eval_results = None
    for i in range(len(divided_images)):
        modelDir = os.path.dirname(os.path.realpath(__file__)) \
                + "/model" + name + "-" + str(i+1) + "_of_" + str(nFolds)
        classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=modelDir)
   
        train_data, eval_data = dataUtils.prepare_training_test_data(divided_images, i, np.float16)
        train_labels, eval_labels = dataUtils.prepare_training_test_data(divided_labels, i, np.int32)
   
        if doTrain:
            train(classifier, train_data, train_labels)
        eval_results = test(classifier, eval_data, eval_labels)
        if all_eval_results:
            for key in eval_results.keys():
                all_eval_results[key] += eval_results[key]
        else:
            all_eval_results = eval_results
    
    for key in all_eval_results.keys():
        all_eval_results[key] = all_eval_results[key]/nFolds
    print("AVERAGE RESULTS {}{}-fold\n{}".format(name, nFolds,all_eval_results))

def predict(classifier, data):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data},
        y=data,
        shuffle=False
    )
    prediction = classifier.predict(input_fn=predict_input_fn)
    print(list(prediction))



def experimentOurVideos(name = ""):
    
#    dataTrainingDir = "ucf-101"
#    dataTestingDir = "own",
#    trainingDir = "our_training"
#    testingDir = "our_testing"
#    imgUtils.createData(dataTrainingDir, trainingDir, False, 1)
#    imgUtils.createData(dataTestingDir, testingDir, True, 10)
    
    if (name != ""):
        name = "_" + name
    modelDir = os.path.dirname(os.path.realpath(__file__)) \
                + "/model" + name + "testOurVideos"
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=modelDir)
    
    train_data, train_labels, eval_data, eval_labels = imgUtils.prepareTestOurVideos()
    
    train(classifier, train_data, train_labels)
    test(classifier, eval_data, eval_labels)
    print(eval_data)
    
    
def main(argv):
    images, labels = imgUtils.processVideos()

    ### BEGIN RUN ONCE ###
#
    modelDir = os.path.dirname(os.path.realpath(__file__)) + "/model"
    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=modelDir)
#    classifier = tf.estimator.Estimator(model_fn=cnn_4_layer_model_fn, model_dir=modelDir+"_4layer")
#    classifier = tf.estimator.Estimator(model_fn=cnn_9x9_model_fn, model_dir=modelDir+"_9x9kernel")
#
#    train_data, train_labels, eval_data, eval_labels = dataUtils.simple_split(images, labels)
#
#    train(classifier, train_data, train_labels)
#    test(classifier, eval_data, eval_labels)

    train_data, train_labels, eval_data, eval_labels = dataUtils.simple_split(images, labels)
#    predict(classifier, train_data[0]) # Give the classification for a single frame

    train(classifier, train_data, train_labels) # Train the CNN on all frames in the training folder
    test(classifier, eval_data, eval_labels)

    ### END RUN ONCE ###
    
#    experimentOurVideos()
    

    ### BEGIN 10 CROSSFOLD VALIDATION ###

#    nFolds = 2
#    name = "{}foldcv".format(nFolds)
#    experiment_n_fold(images, labels, nFolds, name, False)
#    
#    nFolds = 3
#    name = "{}foldcv".format(nFolds)
#    experiment_n_fold(images, labels, nFolds, name, True)
#
#    nFolds = 10
#    name = "base".format(nFolds)
#    experiment_n_fold(images, labels, nFolds, name, False)
#    
#    nFolds = 5
#    name = "{}foldcv".format(nFolds)
#    experiment_n_fold(images, labels, nFolds, name, False)

    ### END 10 CROSSFOLD VALIDATION ###


if __name__ == "__main__":
    #imgUtils.createData(flip = False, multiple = 1)
    #imgUtils.createData(dataDir = "own", flip = False, multiple = 1)
    tf.app.run()