from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
import tensorflow as tf

from src.main.python.preprocessing.label_encoding import num2class
from src.main.python.preprocessing.normalization import normalize_image


def keras_classify_sample_img(path, target_size, model_name):
    """Classify sample image with keras model

    Args:
        path (str): String path to image
        target_size (tuple): Tuple containing image shape as width, height, channels
        model_name (str): Name of a Keras model to use
    Returns:
        prediction (List(str)): predicted classes
    """

    model = load_model("../../resources/models/{model_name}.h5".format(model_name=model_name))

    sample_img = load_img(path, target_size=target_size)
    sample_img_tensor = np.expand_dims(normalize_image(sample_img), axis=0)

    prediction = num2class(model.predict_classes(sample_img_tensor, verbose=0))

    return prediction


def tf_classify_sample_img(path, target_size, model_name):
    """Classify sample image with tf.core model

    Args:
        path (str): String path to image
        target_size (tuple): Tuple containing image shape as width, height, channels
        model_name (str): Name of a TensorFlow model to use
    Returns:
        prediction (List(str)): predicted classes
    """

    sess = tf.Session()
    saver = tf.train.import_meta_graph("../../resources/models/{model_name}".format(model_name=model_name))
    saver.restore(sess, tf.train.latest_checkpoint("../../resources/models"))

    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("X:0")

    predict = graph.get_tensor_by_name("predict:0")

    sample_img = load_img(path, target_size=target_size)
    sample_img_tensor = np.expand_dims(normalize_image(sample_img), axis=0)

    prediction = num2class(sess.run(predict, feed_dict={X: sample_img_tensor}))

    return prediction


if __name__ == "__main__":

    image_width, image_height, channels = 480, 270, 3
    model_name = "keras_classification_cnn"
    # model_name = "model.ckpt.meta"

    classify_sample_img = keras_classify_sample_img if model_name.startswith("keras") else tf_classify_sample_img

    sample_images = [
        "BTR-80/20190714140917_1.jpg",
        "T-55/20190714140241_1.jpg",
        "BMP-1/20190718163640_1.jpg",
        "T-72B/20190720165103_1.jpg",
        "Background/20190902185121_1.jpg"
    ]

    for img_path in sample_images:
        pred = classify_sample_img(
            path="../../resources/Litening_images/test/{img_path}".format(img_path=img_path),
            target_size=(image_width, image_height, channels),
            model_name=model_name,
        )

        print("Classified image on path {0} as: {1}".format(img_path, pred[0]))
