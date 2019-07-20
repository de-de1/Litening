from keras.models import load_model
import tensorflow as tf

if __name__ == "__main__":
    PATH = "../../resources/models/basic_cnn.h5"
    IMAGE_HEIGHT, IMAGE_WIDTH = 480, 270
    BATCH_SIZE = 16

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    test_data_generator = datagen.flow_from_directory("../../resources/Litening_images/test",
                                                       target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE)

    model = load_model(PATH)

    loss, acc = model.evaluate_generator(test_data_generator, steps=5)
    print(loss)
    print(acc)
