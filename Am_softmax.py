# -*- coding: utf-8 -*-
"""
AM-Softmax 通过增加margin来提高分类的界面，相比Softmax有很大的提升空间
"""
from keras.engine.topology import Layer
from keras.layers import initializers, regularizers, constraints
from keras.engine import InputSpec
import tensorflow as tf
import keras.backend as K
from keras.optimizers import SGD

sgd = SGD()
from keras.preprocessing.image import ImageDataGenerator

ImageDataGenerator.flow_from_directory()


class AMSoftmax(Layer):

    def __init__(self, units, s, m,
                 kernel_initializer='glorot_uniform',  # 初始化方式文章中给的
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMSoftmax, self).__init__(**kwargs)  # 继承来自keras的Layer
        self.units = units
        self.s = s
        self.m = m
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param inputs: input_tensor
        :param kwargs: 继承父类的dict
        :return: amsoftmax = exp(s*[(w.t) - m])(j) / exp(s*[(w.t) - m])(j) + exp(s*[(w.t)](k != i)) output_tensor
        """
        # 把weights 和 x都做norm
        inputs = tf.nn.l2_normalize(inputs, dim=-1)  # 对x做归一化
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=(0, 1))   # 对W归一化 |w|.|x| = 1

        print('amsoftmax norm inputs:', inputs)
        print('amsoftmax norm self.kernel:', self.kernel)
        K.l2_normalize()

        cos_theta = tf.matmul(inputs, self.kernel)  # cos(theta) = norm(W).norm(x)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)  # for normalize stead
        psi = cos_theta - self.m  # amsoftmax margin: cos(theta) - m

        e_costheta = K.exp(self.s * cos_theta)  # 其他类e(s. cos_theta)
        e_psi = K.exp(self.s * psi)  # 待分类e(s. (cos_theta - m))
        sum_x = K.sum(e_costheta, axis=-1, keepdims=True)  # 求其他类的和

        temp = e_psi - e_costheta
        temp = temp + sum_x  # 待分类与未分类的和

        amsoftmax = e_psi / temp  # amsoftmax
        return amsoftmax


# AMSoftmax loss function
def amsoftmax_loss(y_true, y_pred):
    """
    :param y_true: real label
    :param y_pred: predict label
    :return: cross entropy of amsoftmax
    """
    # d1 = K.sum(y_true * y_pred, axis=-1)
    # d1 = K.log(K.clip(d1, K.epsilon(), None))
    # loss = -K.mean(d1, axis=-1)
    loss = K.categorical_crossentropy(y_true, y_pred)

    return loss


