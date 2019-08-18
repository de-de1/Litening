import tensorflow as tf


def random_normal_init(shape):
    """Create Variable initialized with random normal distribution with given shape

    Args:
        shape (List): List of tensor target shape
    Returns:
        tf.Variable: initialized with random normal distribution
    """

    return tf.Variable(tf.random_normal(shape, stddev=0.1))


if __name__ == "__main__":
    pass