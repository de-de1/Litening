import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.preprocessing import OneHotEncoder


if __name__ == "__main__":
    EPOCHS = 6
    BATCH_SIZE = 16
    IMAGE_WIDTH = 480
    IMAGE_HEIGHT = 270
    CHANNELS = 3
    NUM_CLASSES = 4

    classes_dict = {"BMP-1": 0, "BTR-80": 1, "T-55": 2, "T-72B": 3}
    classes = np.array([[classes_dict["BMP-1"]], [classes_dict["BTR-80"]], [classes_dict["T-55"]],
                        [classes_dict["T-72B"]]])

    enc = OneHotEncoder()
    enc.fit(classes)

    all_files_path = glob.glob("../../resources/Litening_images/train/*/*")
    random.shuffle(all_files_path)

    num_examples = len(all_files_path)

    for epoch in range(EPOCHS):

        for batch_num in range(num_examples // BATCH_SIZE + 1):

            batch = all_files_path[batch_num * BATCH_SIZE:(batch_num + 1) * BATCH_SIZE]

            X_train_batch = np.ndarray(shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
            Y_train_batch = np.ndarray(shape=(BATCH_SIZE, NUM_CLASSES))

            for idx, file_path in enumerate(batch):
                img = load_img(file_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
                img = img_to_array(img)
                img /= 255.0

                img_class = file_path.split("\\")[-2]

                X_train_batch[idx] = img
                Y_train_batch[idx] = enc.transform(np.array([[classes_dict[img_class]]])).toarray()
