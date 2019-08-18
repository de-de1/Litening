from keras.models import load_model
import tensorflow as tf


def keras_evaluate_test_set(model, testset_path, shape, batch_size):
    """Evaluate test set on a given model

    Args:
        model (keras.models.Sequential): Keras model
        testset_path (str): Path to test set
        shape (tuple): Tuple of image width and height
        batch_size (int): Batch size on test set images
    Returns:
        loss (float): Loss on test set
        acc (float): Accuracy on test set
    """

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    test_data_generator = datagen.flow_from_directory(testset_path,
                                                      target_size=shape, batch_size=batch_size)
    loss, acc = model.evaluate_generator(test_data_generator, steps=5)

    return loss, acc


if __name__ == "__main__":
    MODEL_PATH = "../../resources/models/basic_cnn.h5"
    TESTSET_PATH = "../../resources/Litening_images/test"
    SHAPE = (480, 270)
    BATCH_SIZE = 16

    model = load_model(MODEL_PATH)

    loss, acc = keras_evaluate_test_set(model, TESTSET_PATH, SHAPE, BATCH_SIZE)
    print(loss)
    print(acc)
