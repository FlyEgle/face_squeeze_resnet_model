"""
This python script is for model build
"""
from __future__ import print_function
from keras.models import Model, Input
from keras.layers import concatenate
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
import keras.backend as K

K.set_image_data_format('channels_last')


# SqueezeNet
def firenet(output_channel1, output_channel2, above_layer):

    squeeze = Conv2D(output_channel1, (1, 1), activation='relu', padding='same')(above_layer)
    expand_1 = Conv2D(output_channel2, (1, 1), activation='relu', padding='same')(squeeze)
    expand_2 = Conv2D(output_channel2, (1, 1), activation='relu', padding='same')(squeeze)
    expand = concatenate([expand_1, expand_2], axis=-1)
    return expand


def squeezenet(input_shape, out_dims):

    input_image = Input(input_shape)
    conv1 = Conv2D(96, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_image)
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

    fire2 = firenet(16, 64, maxpool1)
    fire3 = firenet(16, 64, fire2)
    fire4 = firenet(32, 128, fire3)

    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire4)

    fire5 = firenet(32, 128, maxpool4)
    fire6 = firenet(48, 192, fire5)
    fire7 = firenet(48, 192, fire6)
    fire8 = firenet(64, 256, fire7)

    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(fire8)

    fire9 = firenet(64, 256, maxpool8)
    conv10 = Conv2D(1024, (1, 1), strides=(1, 1), activation='relu')(fire9)
    bn_1 = BatchNormalization()(conv10)
    avgpool = AveragePooling2D(pool_size=(13, 13), strides=(1, 1))(bn_1)
    flatten = Flatten()(avgpool)
    dp = Dropout(0.7)(flatten)
    fc = Dense(out_dims, activation='softmax')(dp)

    squeeze_model = Model(inputs=input_image, outputs=fc)

    return squeeze_model


def resnet(weights):
    restnet_model = ResNet50(weights=weights, input_shape=(224, 224, 3), classes=529)
    return restnet_model


