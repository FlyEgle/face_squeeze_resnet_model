import numpy as np
from keras import backend as K
from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Dropout,BatchNormalization, PReLU, Flatten, Embedding
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import activations, initializers, regularizers, constraints, Lambda
from keras.applications.resnet50 import ResNet50
import pandas as pd
import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
import keras

np.random.seed(1337)


class AMSoftmax(Layer):
    def __init__(self, units, s, m,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMSoftmax, self).__init__(**kwargs)
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
        inputs = tf.nn.l2_normalize(inputs, dim=-1)
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=(0, 1))   # W归一化

        dis_cosin = K.dot(inputs, self.kernel)
        psi = dis_cosin - self.m

        e_costheta = K.exp(self.s * dis_cosin)
        e_psi = K.exp(self.s * psi)
        sum_x = K.sum(e_costheta, axis=-1, keepdims=True)

        temp = e_psi - e_costheta
        temp = temp + sum_x

        output = e_psi / temp
        return output


# 定义的AMSoftmax_loss 损失函数
def amsoftmax_loss(y_true, y_pred):
    d1 = K.sum(y_true * y_pred, axis=-1)
    d1 = K.log(K.clip(d1, K.epsilon(), None))
    loss = -K.mean(d1, axis=-1)
    return loss


batch_size = 512
num_classes = 10
epochs = 20
feature_size = 128

inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3))(inputs)
x = PReLU()(x)
x = Conv2D(32, (3, 3))(x)
x = PReLU()(x)
x = Conv2D(64, (3, 3))(x)
x = PReLU()(x)
x = Conv2D(64, (5, 5))(x)
x = PReLU()(x)
x = Conv2D(128, (5, 5))(x)
x = PReLU()(x)
x = Conv2D(128, (5, 5))(x)
x = PReLU()(x)
x = Flatten()(x)
x = Dense(feature_size)(x)
ip1 = PReLU(name='ip1')(x)
ip1 = Dropout(0.2)(ip1)
# 这一层是AMSoftmax layer代替传统的FC全连接层
output = AMSoftmax(10, 10, 0.35)(ip1)
# model = Model(inputs=inputs, outputs=output)
# print(model.summary())
# plot_model(model, to_file='AMSoftmax.png')
lambda_c = 0.01
input_target = Input(shape=(1, ))  # single value ground truth labels as inputs
centers = Embedding(num_classes, feature_size)(input_target)
l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([ip1, centers])

model_centerloss_train = Model(inputs=[inputs, input_target], outputs=[output, l2_loss])
model_centerloss_predict = Model(inputs=[inputs], outputs=[output])

print(model_centerloss_train.summary())
plot_model(model_centerloss_train, to_file='AMSoftmax-centerloss.png')


sorted()
