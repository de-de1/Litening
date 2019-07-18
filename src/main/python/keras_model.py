import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.client import device_lib


def cnn_model(shape, no_classes):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=shape
    ))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu'
    ))

    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dense(units=no_classes, activation='softmax'))
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def get_number_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print("Variable shape: {}".format(shape))
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        print("Variable number of parameters: {}".format(variable_parameters))
        total_parameters += variable_parameters
    print(total_parameters)


def predict_sample_img(path, target_size):
    sample_img = load_img(path, target_size=target_size)
    sample_img_tensor = img_to_array(sample_img)
    sample_img_tensor = np.expand_dims(sample_img_tensor, axis=0)
    sample_img_tensor /= 255.

    cnn_prediction = num2class(simple_cnn_model.predict_classes(
        sample_img_tensor,
        verbose=0))

    return cnn_prediction


def num2class(num_classes):
    str_classes = []
    for x in num_classes:
        if x == 0:
            str_classes.append('bmp-1')
        elif x == 1:
            str_classes.append('btr-80')
        else:
            str_classes.append('t-55')
    return str_classes


def class2num(str_classes):
    return [0 if x == 'btr-80' else 1 for x in str_classes]


if __name__ == "__main__":
    print(device_lib.list_local_devices())
    K.tensorflow_backend._get_available_gpus()

    BATCH_SIZE = 16
    NUM_CLASSES = 3
    EPOCHS = 6
    image_height, image_width = 320, 240

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_data_generator = train_datagen.flow_from_directory("../resources/Litening_images/train",
                                                       target_size=(image_height, image_width), batch_size=BATCH_SIZE)

    test_data_generator = test_datagen.flow_from_directory("../resources/Litening_images/test",
                                                             target_size=(image_height, image_width),
                                                             batch_size=BATCH_SIZE)

    input_shape = (image_height, image_width, 3)

    simple_cnn_model = cnn_model(input_shape, NUM_CLASSES)

    simple_cnn_model.fit_generator(train_data_generator, epochs=EPOCHS, validation_data=test_data_generator)

    print(
        predict_sample_img('../resources/Litening_images/test/BTR-80/20190714140607_1.jpg', (image_height, image_width)))
    print(
        predict_sample_img('../resources/Litening_images/test/BTR-80/20190714140939_1.jpg', (image_height, image_width)))
    print(
        predict_sample_img('../resources/Litening_images/test/T-55/20190714140244_1.jpg', (image_height, image_width)))
    print(
        predict_sample_img('../resources/Litening_images/test/T-55/20190714140415_1.jpg', (image_height, image_width)))
    print(
        predict_sample_img('../resources/Litening_images/test/BMP-1/20190718163429_1.jpg', (image_height, image_width)))
    print(
        predict_sample_img('../resources/Litening_images/test/BMP-1/20190718163450_1.jpg', (image_height, image_width)))
