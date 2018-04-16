# -*- coding: utf-8 -*-
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Activation,BatchNormalization
from keras.layers import activations, initializers, regularizers, constraints, Lambda
from keras.engine import InputSpec
import tensorflow as tf
import numpy as np


class ASoftmax(Dense):
    def __init__(self, units, m, batch_size,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.m = m
        self.batch_size = batch_size
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


    def call(self, inputs):
        inputs.set_shape([self.batch_size, inputs.shape[-1]])
        inputs_norm = K.sqrt(K.sum(K.square(inputs), axis=-1, keepdims=True))
        kernel_norm = tf.nn.l2_normalize(self.kernel, dim=(0, 1))                          # W归一化
        inner_product = K.dot(inputs, kernel_norm)  # w*x
        dis_cosin = inner_product / inputs_norm

        m_cosin = multipul_cos(dis_cosin, self.m)
        sum_y = K.sum(K.exp(inputs_norm * dis_cosin), axis=-1, keepdims=True)  # sum（exp(w*X)）
        k = get_k(dis_cosin, self.units, self.batch_size)
        psi = np.power(-1, k) * m_cosin - 2 * k
        e_x = K.exp(inputs_norm * dis_cosin)
        e_y = K.exp(inputs_norm * psi)
        sum_x = K.sum(e_x, axis=-1, keepdims=True)
        temp = e_y - e_x
        temp = temp + sum_x

        output = e_y / temp
        return output


def multipul_cos(x, m):
    if m == 2:
        x = 2 * K.pow(x, 2) - 1
    elif m == 3:
        x = 4 * K.pow(x, 3) - 3 * x
    elif m == 4:
        x = 8 * K.pow(x, 4) - 8 * K.pow(x, 2) + 1
    else:
        raise ValueError("To high m")
    return x


def get_k(m_cosin, out_num, batch_num):
    theta_yi = tf.acos(m_cosin)  #[0,pi]
    theta_yi = tf.reshape(theta_yi, [-1])
    pi = K.constant(3.1415926)

    def cond(p1, p2, k_temp, theta):
        return K.greater_equal(theta, p2)

    def body(p1, p2, k_temp, theta):
        k_temp += 1
        p1 = k_temp * pi / out_num
        p2 = (k_temp + 1) * pi / out_num
        return p1, p2, k_temp, theta

    k_list = []
    for i in range(batch_num * out_num):
        k_temp = K.constant(0)
        p1 = k_temp * pi / out_num
        p2 = (k_temp + 1) * pi / out_num
        _, _, k_temp, _ = tf.while_loop(cond, body, [p1, p2, k_temp, theta_yi[i]])
        k_list.append(k_temp)
    k = K.stack(k_list)
    k = tf.squeeze(K.reshape(k, [batch_num, out_num]))
    return k


def asoftmax_loss(y_true, y_pred):
    d1 = K.sum(tf.multiply(y_true, y_pred), axis=-1)
    p = -K.log(d1)
    loss = K.mean(p)
    K.print_tensor(loss)
    return p
