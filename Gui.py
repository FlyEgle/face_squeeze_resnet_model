import tensorflow as tf
import keras
import numpy as np
import keras.backend as K
from keras.layers import BatchNormalization
from keras.models import Model, Input


def GroupNorm(x, gamma, beta, G, eps=1e-5):

    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C//G, H, W])

    print(x.shape)

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    print(mean, var)
    x = (x - mean) / tf.sqrt(var + eps)

    x = tf.reshape(x, [N, C, H, W])

    return x *gamma + beta


x = np.random.random((32, 16, 28, 28))
x = GroupNorm(x, gamma=[1, 16, 1, 1], beta=[1, 16, 1, 1], G=2)
# print(x.shape)

