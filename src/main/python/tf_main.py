import tensorflow as tf
import numpy as np
import glob
import random
from keras.preprocessing.image import load_img
from tensorflow.python.client import device_lib
from preprocessing.normalization import normalize_image
from initialization.init_methods import random_normal_init
from initialization.label_encoding import encode
from model.tf_model import model


if __name__ == "__main__":
    EPOCHS = 1
    BATCH_SIZE = 16
    IMAGE_WIDTH = 480
    IMAGE_HEIGHT = 270
    CHANNELS = 3
    NUM_CLASSES = 4
    TRAIN_FILES_PATH = "../resources/Litening_images/train/*/*"
    CLASSES_DICT = {"BMP-1": 0, "BTR-80": 1, "T-55": 2, "T-72B": 3}

    classes_encoded = np.array([[CLASSES_DICT["BMP-1"]], [CLASSES_DICT["BTR-80"]], [CLASSES_DICT["T-55"]],
                        [CLASSES_DICT["T-72B"]]])

    encoder = encode(classes_encoded)

    all_files_paths = glob.glob(TRAIN_FILES_PATH)
    random.shuffle(all_files_paths)

    num_examples = len(all_files_paths)

    X = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
    Y = tf.placeholder(dtype=tf.float32, shape=(None, NUM_CLASSES))

    w1 = random_normal_init(shape=[3, 3, 3, 32])
    w2 = random_normal_init(shape=[3, 3, 32, 32])
    w3 = random_normal_init(shape=[3, 3, 32, 64])
    w4 = random_normal_init(shape=[3, 3, 64, 64])
    w_o = random_normal_init(shape=[64 * 117 * 64, NUM_CLASSES])

    feed = model(X, w1, w2, w3, w4, w_o)

    Y_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=feed)

    predict_op = tf.argmax(model(X, w1, w2, w3, w4, w_o), 1)
    predict = model(X, w1, w2, w3, w4, w_o)
    predict_softmax = tf.nn.softmax(model(X, w1, w2, w3, w4, w_o))

    cost = tf.reduce_mean(Y_)

    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter('./logs/1/train', sess.graph)

        for epoch in range(EPOCHS):

            for batch_num in range(num_examples // BATCH_SIZE + 1):

                batch = all_files_paths[batch_num * BATCH_SIZE:(batch_num + 1) * BATCH_SIZE]

                X_train_batch = np.ndarray(shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
                Y_train_batch = np.ndarray(shape=(BATCH_SIZE, NUM_CLASSES))

                for idx, file_path in enumerate(batch):
                    img = load_img(file_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
                    normalized_img = normalize_image(img)

                    img_class = file_path.split("\\")[-2]

                    X_train_batch[idx] = normalized_img
                    Y_train_batch[idx] = encoder.transform(np.array([[CLASSES_DICT[img_class]]])).toarray()

                batch_cost, _ = sess.run([cost, optimizer], feed_dict={X: X_train_batch, Y: Y_train_batch})
                print(epoch, batch_num, batch_cost)

        X_sample = np.ndarray(shape=(4, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
        path1 = "C:\\Users\\Phantom\\Desktop\\Python\\development\\Litening\\src\\main\\resources\\Litening_images\\test\\BMP-1\\20190718163424_1.jpg"
        path2 = "C:\\Users\\Phantom\\Desktop\\Python\\development\\Litening\\src\\main\\resources\\Litening_images\\test\\BTR-80\\20190714140606_1.jpg"
        path3 = "C:\\Users\\Phantom\\Desktop\\Python\\development\\Litening\\src\\main\\resources\\Litening_images\\test\\T-55\\20190714140414_1.jpg"
        path4 = "C:\\Users\\Phantom\\Desktop\\Python\\development\\Litening\\src\\main\\resources\\Litening_images\\test\\T-72B\\20190720165103_1.jpg"
        # img1 = image_load(path1, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
        # img2 = image_load(path2, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
        # img3 = image_load(path3, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
        # img4 = image_load(path4, (IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
        # X_sample[0] = img1
        # X_sample[1] = img2
        # X_sample[2] = img3
        # X_sample[3] = img4

        print(sess.run(predict, feed_dict={X: X_sample}))
        print(sess.run(predict_op, feed_dict={X: X_sample}))
        print(sess.run(predict_softmax, feed_dict={X: X_sample}))

        save_path = saver.save(sess, "../../resources/models/model.ckpt")
