import tensorflow as tf


def model(X, w1, w2, w3, w4, w_o):
    """Create tensorflow core model for classification

    Args:
        X (tf.placeholder): input X feed
        w1 (tf.Variable): weights of convolutional layer 1
        w2 (tf.Variable): weights of convolutional layer 2
        w3 (tf.Variable): weights of convolutional layer 3
        w4 (tf.Variable): weights of convolutional layer 4
        w_o (tf.Variable): weights of fully connected output layer
    Returns:
        fc_layer: logits of last layer
    """

    conv1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding="VALID")
    conv1_activation = tf.nn.relu(conv1)
    conv2 = tf.nn.conv2d(conv1_activation, w2, strides=[1, 1, 1, 1], padding="VALID")
    conv2_activation = tf.nn.relu(conv2)
    conv2_pooling = tf.nn.max_pool(conv2_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    conv3 = tf.nn.conv2d(conv2_pooling, w3, strides=[1, 1, 1, 1], padding="VALID")
    conv3_activation = tf.nn.relu(conv3)
    conv4 = tf.nn.conv2d(conv3_activation, w4, strides=[1, 1, 1, 1], padding="VALID")
    conv4_activation = tf.nn.relu(conv4)
    conv4_pooling = tf.nn.max_pool(conv4_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    flatten = tf.reshape(conv4_pooling, shape=[-1, w_o.get_shape().as_list()[0]])
    fc_layer = tf.matmul(flatten, w_o)

    return fc_layer


if __name__ == "__main__":
    pass
