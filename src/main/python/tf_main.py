import tensorflow as tf
import numpy as np
import glob
import random
from keras.preprocessing.image import load_img
from src.main.python.preprocessing.normalization import normalize_image
from src.main.python.initialization.init_methods import random_normal_init
from src.main.python.preprocessing.label_encoding import encode
from src.main.python.model.tf_model import model


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

    X = tf.placeholder(dtype=tf.float32, shape=(None, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS), name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=(None, NUM_CLASSES), name="Y")

    w1 = random_normal_init(shape=[3, 3, 3, 32])
    w2 = random_normal_init(shape=[3, 3, 32, 32])
    w3 = random_normal_init(shape=[3, 3, 32, 64])
    w4 = random_normal_init(shape=[3, 3, 64, 64])
    w_o = random_normal_init(shape=[64 * 117 * 64, NUM_CLASSES])

    feed = model(X, w1, w2, w3, w4, w_o)

    Y_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=feed)

    predict_op = tf.argmax(model(X, w1, w2, w3, w4, w_o), 1, name="predict")
    predict = model(X, w1, w2, w3, w4, w_o)
    predict_softmax = tf.nn.softmax(model(X, w1, w2, w3, w4, w_o), name="predict_softmax")

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

        save_path = saver.save(sess, "../resources/models/model.ckpt")
