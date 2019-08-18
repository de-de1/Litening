import numpy as np
from keras.preprocessing.image import load_img
from keras.models import load_model
from src.main.python.preprocessing.normalization import normalize_image
from src.main.python.initialization.label_encoding import num2class


def keras_predict_sample_img(path, target_size, model):
    """Predict sample image on a given path

    Args:
        path (str): String path to image
        target_size (tuple): Tuple containing image shape
        model (keras.models.Sequential): Keras model
    Returns:
        cnn_prediction (List(str)): predicted classes
    """

    sample_img = load_img(path, target_size=target_size)
    sample_img_tensor = np.expand_dims(normalize_image(sample_img), axis=0)

    cnn_prediction = num2class(model.predict_classes(sample_img_tensor, verbose=0))

    return cnn_prediction


if __name__ == "__main__":
    model = load_model('../../resources/models/basic_cnn.h5')
    IMAGE_WIDTH, IMAGE_HEIGHT = 480, 270

    print(
        keras_predict_sample_img('../../resources/Litening_images/test/BTR-80/20190714140917_1.jpg',
                           (IMAGE_WIDTH, IMAGE_HEIGHT), model))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/BTR-80/20190718230231_1.jpg',
                           (IMAGE_WIDTH, IMAGE_HEIGHT), model))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/T-55/20190714140241_1.jpg',
                           (IMAGE_WIDTH, IMAGE_HEIGHT), model))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/T-55/20190718224824_1.jpg',
                           (IMAGE_WIDTH, IMAGE_HEIGHT), model))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/BMP-1/20190718163640_1.jpg',
                           (IMAGE_WIDTH, IMAGE_HEIGHT), model))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/BMP-1/20190718164745_1.jpg',
                           (IMAGE_WIDTH, IMAGE_HEIGHT), model))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/T-72B/20190720165103_1.jpg',
                           (IMAGE_WIDTH, IMAGE_HEIGHT), model))
    print(
        keras_predict_sample_img('../../resources/Litening_images/test/T-72B/20190720174144_1.jpg',
                           (IMAGE_WIDTH, IMAGE_HEIGHT), model))
