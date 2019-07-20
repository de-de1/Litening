import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


def num2class(num_classes):
    str_classes = []
    for x in num_classes:
        if x == 0:
            str_classes.append('bmp-1')
        elif x == 1:
            str_classes.append('btr-80')
        elif x == 2:
            str_classes.append('t-55')
        elif x == 3:
            str_classes.append('t-72b')
    return str_classes


def class2num(str_classes):
    num_classes = []
    for x in str_classes:
        if x == 'bmp-1':
            num_classes.append(0)
        elif x == 'btr-80':
            num_classes.append(1)
        elif x == 't-55':
            num_classes.append(2)
        elif x == 't-72b':
            num_classes.append(3)
    return num_classes


def predict_sample_img(path, target_size, model):
    sample_img = load_img(path, target_size=target_size)
    sample_img_tensor = img_to_array(sample_img)
    sample_img_tensor = np.expand_dims(sample_img_tensor, axis=0)
    sample_img_tensor /= 255.

    cnn_prediction = num2class(model.predict_classes(sample_img_tensor, verbose=0))

    return cnn_prediction


if __name__ == "__main__":
    model = load_model('../../resources/models/basic_cnn.h5')
    image_height, image_width = 480, 270

    print(
        predict_sample_img('../../resources/Litening_images/test/BTR-80/20190714140917_1.jpg',
                           (image_height, image_width), model))
    print(
        predict_sample_img('../../resources/Litening_images/test/BTR-80/20190718230231_1.jpg',
                           (image_height, image_width), model))
    print(
        predict_sample_img('../../resources/Litening_images/test/T-55/20190714140241_1.jpg',
                           (image_height, image_width), model))
    print(
        predict_sample_img('../../resources/Litening_images/test/T-55/20190718224824_1.jpg',
                           (image_height, image_width), model))
    print(
        predict_sample_img('../../resources/Litening_images/test/BMP-1/20190718163640_1.jpg',
                           (image_height, image_width), model))
    print(
        predict_sample_img('../../resources/Litening_images/test/BMP-1/20190718164745_1.jpg',
                           (image_height, image_width), model))
    print(
        predict_sample_img('../../resources/Litening_images/test/T-72B/20190720165103_1.jpg',
                           (image_height, image_width), model))
    print(
        predict_sample_img('../../resources/Litening_images/test/T-72B/20190720174144_1.jpg',
                           (image_height, image_width), model))
