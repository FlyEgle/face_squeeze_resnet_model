from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.datasets import mnist
import functools
import keras.backend as K
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np


img_rows, img_cols = 28, 28


def _center_loss_func(features, labels, alpha, num_classes,
                      centers, feature_dim):
    assert feature_dim == features.get_shape()[1]
    labels = K.reshape(labels, [-1])
    labels = tf.to_int32(labels)
    centers_batch = tf.gather(centers, labels)
    diff = (1 - alpha) * (centers_batch - features)
    centers = tf.scatter_sub(centers, labels, diff)
    loss = tf.reduce_mean(K.square(features - centers_batch))
    return loss


def get_center_loss(alpha, num_classes, feature_dim):
    """Center loss based on the paper "A Discriminative
       Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    # Each output layer use one independed center: scope/centers
    centers = K.zeros([num_classes, feature_dim])

    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        return _center_loss_func(y_pred, y_true, alpha,
                                 num_classes, centers, feature_dim)
    return center_loss


def myGenerator():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = to_categorical(y_train,10)
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    while 1:
        for i in range(1875): # 1875 * 32 = 60000 -> # of training samples
            if i % 125 == 0:
                print("i = " + str(i))
            yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]


def clone_y_generator(generator):
    # output: train_gen_X, [train_gen_Y, train_gen_Y]
    while True:
        data = next(generator)
        x = data[0]
        y = [data[1], data[1]]
        yield x, y


# one-hot转换成普通label
def translate_onehot2label(one_hot):

    # length = num of images labels, nb_classes = classes of image
    length = one_hot.shape[0]
    nb_classes = one_hot.shape[1]

    labels = []
    for i in range(length):
        for j in range(nb_classes):
            if one_hot[i][j] == 1:
                labels.append(j)

    labels = np.array(labels).reshape((length, 1))

    return labels


data = next(myGenerator())
label = data[1]
print(label.shape)
print(translate_onehot2label(label).shape)
print(translate_onehot2label(label))

