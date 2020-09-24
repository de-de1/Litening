import math
import os
import random

from keras.layers import Conv2D, Dense, Input, MaxPooling2D, Flatten
from keras.losses import mean_squared_error, binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.utils import Sequence
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.main.python.settings import *
from src.main.python.preprocessing.normalization import normalize_image


class DataGenerator(Sequence):

    def __init__(self, path):
        self.class_dirs = os.listdir(path)
        self.image_file_paths = []
        for dir in self.class_dirs:
            for file in os.listdir(os.path.join(path, dir)):
                if file.endswith("jpg"):
                    self.image_file_paths.append(os.path.join(path, dir, file))
        random.shuffle(self.image_file_paths)
        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        return int(math.ceil(len(self.image_file_paths) / BATCH_SIZE))

    def __getitem__(self, idx):
        batch_paths = self.image_file_paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        batch_coords = np.zeros((BATCH_SIZE, 4))
        batch_classes = np.zeros((BATCH_SIZE, 1))

        for index, image_path in enumerate(batch_paths):
            img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE, 3))
            batch_images[index] = np.expand_dims(normalize_image(img), axis=0)
            img.close()

            if os.path.exists(os.path.splitext(image_path)[0] + ".txt"):
                with open(os.path.splitext(image_path)[0] + ".txt") as annotation_file:
                    row = annotation_file.readline().rstrip().split(" ")
                    batch_classes[index, 0] = int(row[0])
                    batch_coords[index, 0] = float(row[1])
                    batch_coords[index, 1] = float(row[2])
                    batch_coords[index, 2] = float(row[3])
                    batch_coords[index, 3] = float(row[4])
            else:
                batch_classes[index, :] = 0
                batch_coords[index, :] = 0

        return batch_images, {"coords": batch_coords, "classes": batch_classes}

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


def create_model():
    input_layer = Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="input")

    conv_2d_1 = Conv2D(32, 3, activation="relu")(input_layer)
    conv_2d_2 = Conv2D(32, 3, activation="relu")(conv_2d_1)
    max_pool_2 = MaxPooling2D()(conv_2d_2)

    conv_2d_3 = Conv2D(64, 3, activation="relu")(max_pool_2)
    conv_2d_4 = Conv2D(64, 3, activation="relu")(conv_2d_3)
    max_pool_4 = MaxPooling2D()(conv_2d_4)

    flatten = Flatten()(max_pool_4)
    dense_output1 = Dense(4, name="coords")(flatten)
    dense_output2 = Dense(1, name="classes", activation="sigmoid")(flatten)

    return Model(inputs=input_layer, outputs=[dense_output1, dense_output2])


def draw_bounding_box(img, pred_coords, pred_class):
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    x = (pred_coords[0] - (pred_coords[2] / 2)) * 256
    y = (pred_coords[1] - (pred_coords[3] / 2)) * 256
    width = pred_coords[2] * 256
    height = pred_coords[3] * 256

    if pred_class[0] < 0.5:
        class_name = "BMP-1"
    else:
        class_name = "BTR-80"

    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.text(x + 5, y + 5, class_name, color="white")
    plt.show()


if __name__ == "__main__":
    path = "../../resources/localization/train"

    train_datagen = DataGenerator(path)

    model = create_model()

    model.compile(loss=[mean_squared_error, binary_crossentropy], optimizer=Adam(),
                  metrics=[])

    model.fit_generator(generator=train_datagen, epochs=20, steps_per_epoch=len(train_datagen))

    sample_img = load_img("../../resources/localization/train/BMP-1/20190718163453_1.jpg", target_size=(IMAGE_SIZE,
                                                                                                        IMAGE_SIZE, 3))

    sample_img2 = load_img("../../resources/localization/train/BTR-80/20190714140617_1.jpg", target_size=(IMAGE_SIZE,
                                                                                                          IMAGE_SIZE, 3))

    sample_img_tensor = np.expand_dims(normalize_image(sample_img), axis=0)
    sample_img_tensor2 = np.expand_dims(normalize_image(sample_img2), axis=0)

    prediction = model.predict(sample_img_tensor, verbose=0)
    prediction2 = model.predict(sample_img_tensor2, verbose=0)

    print(prediction)
    print(prediction2)

    draw_bounding_box(sample_img, prediction[0][0], prediction[1][0])
    draw_bounding_box(sample_img2, prediction2[0][0], prediction2[1][0])

    # image_file_paths = [os.path.join(path, file) for file in os.listdir(path) if file.endswith("jpg")]
