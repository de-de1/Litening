import glob

from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
import tensorflow as tf

from src.main.python.preprocessing.label_encoding import encode
from src.main.python.preprocessing.normalization import normalize_image


def keras_evaluate_test_set(path, target_size, batch_size):
    """Evaluate test set on keras model

    Args:
        path (str): String path to test set
        target_size (tuple): Tuple containing image shape as width, height
        batch_size (int): Batch size on test set images
    Returns:
        loss (float): Loss on test set
        acc (float): Accuracy on test set
    """

    model = load_model("../../resources/models/basic_cnn.h5")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_data_generator = datagen.flow_from_directory(
        directory=path,
        target_size=target_size,
        batch_size=batch_size
    )

    loss, accuracy = model.evaluate_generator(test_data_generator, steps=5)

    return loss, accuracy


def tf_evaluate_test_set(path, target_size, batch_size):
    """Evaluate test set on tf.core model

    Args:
        path (str): String path to test set
        target_size (tuple): Tuple containing image shape as width, height
        batch_size (int): Batch size on test set images
    Returns:
        loss (float): Loss on test set
        acc (float): Accuracy on test set
    """

    test_set_path = path + "/*/*"

    all_files_paths = glob.glob(test_set_path)

    num_examples = len(all_files_paths)

    CLASSES_DICT = {"BMP-1": 0, "BTR-80": 1, "Background": 2, "T-55": 3, "T-72B": 4}

    classes_encoded = np.array([[CLASSES_DICT["BMP-1"]], [CLASSES_DICT["BTR-80"]], [CLASSES_DICT["Background"]],
                                [CLASSES_DICT["T-55"]], [CLASSES_DICT["T-72B"]]])

    encoder = encode(classes_encoded)

    sess = tf.Session()
    saver = tf.train.import_meta_graph("../../resources/models/model.ckpt.meta")
    saver.restore(sess, tf.train.latest_checkpoint("../../resources/models"))

    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("X:0")

    predict = graph.get_tensor_by_name("predict:0")

    for batch_num in range(num_examples // batch_size + 1):

        batch = all_files_paths[batch_num * batch_size:(batch_num + 1) * batch_size]

        X_test_batch = np.ndarray(shape=(batch_size, *target_size))
        Y_test_batch = np.ndarray(shape=(batch_size, NUM_CLASSES))

        for idx, file_path in enumerate(batch):
            img = load_img(file_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
            normalized_img = normalize_image(img)

            img_class = file_path.split("\\")[-2]

            X_test_batch[idx] = normalized_img
            Y_test_batch[idx] = encoder.transform(np.array([[CLASSES_DICT[img_class]]])).toarray()

        prediction = sess.run(predict, feed_dict={X: X_test_batch})
        print(prediction)


if __name__ == "__main__":
    TESTSET_PATH = "../../resources/Litening_images/test"
    IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS = (480, 270, 3)
    NUM_CLASSES = 5
    BATCH_SIZE = 16

    loss, acc = keras_evaluate_test_set(TESTSET_PATH, (IMAGE_WIDTH, IMAGE_HEIGHT), BATCH_SIZE)
    print(loss)
    print(acc)

    tf_evaluate_test_set(TESTSET_PATH, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), BATCH_SIZE)
