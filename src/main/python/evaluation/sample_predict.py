import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
from src.main.python.preprocessing.label_encoding import num2class
from src.main.python.preprocessing.normalization import normalize_image
import tensorflow as tf


def keras_predict_sample_img(path, target_size):
    """Predict sample image with keras model

    Args:
        path (str): String path to image
        target_size (tuple): Tuple containing image shape as width, height, channels
    Returns:
        prediction (List(str)): predicted classes
    """

    model = load_model('../../resources/models/basic_cnn.h5')

    sample_img = load_img(path, target_size=target_size)
    sample_img_tensor = np.expand_dims(normalize_image(sample_img), axis=0)

    prediction = num2class(model.predict_classes(sample_img_tensor, verbose=0))

    return prediction


def tf_predict_sample_img(path, target_size):
    """Predict sample image with tf.core model

    Args:
        path (str): String path to image
        target_size (tuple): Tuple containing image shape as width, height, channels
    Returns:
        prediction (List(str)): predicted classes
    """

    sess = tf.Session()
    saver = tf.train.import_meta_graph("../../resources/models/model.ckpt.meta")
    saver.restore(sess, tf.train.latest_checkpoint("../../resources/models"))

    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("X:0")

    predict = graph.get_tensor_by_name("predict:0")

    sample_img = load_img(path, target_size=target_size)
    sample_img_tensor = np.expand_dims(normalize_image(sample_img), axis=0)

    prediction = num2class(sess.run(predict, feed_dict={X: sample_img_tensor}))

    return prediction


if __name__ == "__main__":

    IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS = 480, 270, 3

    print(
        keras_predict_sample_img('../../resources/Litening_images/test/BTR-80/20190714140917_1.jpg',
                                 (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/T-55/20190714140241_1.jpg',
                                 (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/BMP-1/20190718163640_1.jpg',
                                 (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/T-72B/20190720165103_1.jpg',
                                 (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/Background/20190902185121_1.jpg',
                                 (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))

    path = "..\\..\\resources\\Litening_images\\test\\BMP-1\\20190718163424_1.jpg"
    path2 = "..\\..\\resources\\Litening_images\\test\\BTR-80\\20190714140606_1.jpg"
    path3 = "..\\..\\resources\\Litening_images\\test\\T-55\\20190714140414_1.jpg"
    path4 = "..\\..\\resources\\Litening_images\\test\\T-72B\\20190720165103_1.jpg"
    path5 = "..\\..\\resources\\Litening_images\\test\\Background\\20190902185333_1.jpg"

    print(tf_predict_sample_img(path, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
    print(tf_predict_sample_img(path2, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
    print(tf_predict_sample_img(path3, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
    print(tf_predict_sample_img(path4, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))),
    print(tf_predict_sample_img(path5, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
