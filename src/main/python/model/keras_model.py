import tensorflow as tf
from keras import backend as K
from tensorflow.python.client import device_lib


def cnn_model(shape, no_classes):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

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


if __name__ == "__main__":
    print(device_lib.list_local_devices())
    K.tensorflow_backend._get_available_gpus()

    BATCH_SIZE = 16
    NUM_CLASSES = 4
    EPOCHS = 6
    image_height, image_width = 480, 270

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_data_generator = datagen.flow_from_directory("../resources/Litening_images/train",
                                                       target_size=(image_height, image_width), batch_size=BATCH_SIZE)

    validation_data_generator = datagen.flow_from_directory("../resources/Litening_images/validation",
                                                             target_size=(image_height, image_width),
                                                             batch_size=BATCH_SIZE)

    input_shape = (image_height, image_width, 3)

    simple_cnn_model = cnn_model(input_shape, NUM_CLASSES)

    simple_cnn_model.summary()

    simple_cnn_model.fit_generator(train_data_generator, epochs=EPOCHS, validation_data=validation_data_generator)

    simple_cnn_model.save("../resources/models/basic_cnn.h5")
