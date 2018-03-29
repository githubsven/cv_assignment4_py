import tensorflow as tf
import os
import imgUtils, dataUtils


#Edit paper: https://www.sharelatex.com/5152532135sdnhgphhqsbd

tf.logging.set_verbosity(tf.logging.INFO)


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
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 86, 86, 32]
  # Output Tensor Shape: [batch_size, 43, 43, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=4)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 19, 19, 64]
  # Output Tensor Shape: [batch_size, 19 * 19 * 64]
  pool2_flat = tf.reshape(pool1, [-1, 23 * 23 * 32])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 19 * 19 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

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
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train(classifier, data, labels):
    """
    TODO Train the NN
    """
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=5)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data},
        y=labels,
        batch_size=64,
        num_epochs=3,
        shuffle=False)

    classifier.train(
        input_fn=train_input_fn,
        steps=2000,
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
    print(eval_results)


def main(argv):
    images, labels = imgUtils.processVideos()

    ### BEGIN RUN ONCE ###

    #modelDir = os.path.dirname(os.path.realpath(__file__)) + "/model"
    #classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=modelDir)

    # train_data, train_labels, eval_data, eval_labels = simple_split(images, labels)

    #train(classifier, train_data, train_labels)
    #test(classifier, eval_data, eval_labels)

    ### END RUN ONCE ###


    ### BEGIN 10 CROSSFOLD VALIDATION ###

    divided_images = dataUtils.divide_data(images)
    divided_labels = dataUtils.divide_data(labels)

    for i in range(len(divided_images)):
        modelDir = os.path.dirname(os.path.realpath(__file__)) + "/model_" + str(i)
        classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=modelDir)

        train_data, eval_data = dataUtils.prepare_training_test_data(divided_images, i)
        train_labels, eval_labels = dataUtils.prepare_training_test_data(divided_labels, i)

        train(classifier, train_data, train_labels)
        test(classifier, eval_data, eval_labels)

    ### END 10 CROSSFOLD VALIDATION ###


if __name__ == "__main__":
    #imgUtils.createData(flip = True, multiple = 10)
    #imgUtils.createData(dataDir = "own", flip = True, multiple = 10)
    tf.app.run()