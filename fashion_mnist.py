from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten, Dense, BatchNormalization, PReLU, Dropout, concatenate, UpSampling2D
from keras.models import Model, Input
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.layers import Conv2DTranspose, Cropping2D
import keras.backend as K
from keras.metrics import top_k_categorical_accuracy
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# input_shape = Input(shape=input_shape)
# x = Conv2D(32, (3, 3), activation='relu')(input_shape)
# x = Conv2D(32, (3, 3), activation='relu')(x)
# x = MaxPooling2D((2, 2))(x)
# x = Flatten()(x)
# x = Dense(64, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(0.2)(x)
# x = Dense(128, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(0.2)(x)
# x = Dense(10, activation='softmax')(x)
# model = Model(inputs=input_shape, outputs=x)
#
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))


def build_model(x):

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    return x


def merge_model(input_shape):

    input1 = Input(input_shape)
    input2 = Input(input_shape)
    x1 = build_model(input1)
    x2 = build_model(input2)

    cat = concatenate([x1, x2], axis=-1)
    x = Dense(10, activation='softmax')(cat)

    model = Model(inputs=[input1, input2], outputs=x)
    return model


model = merge_model((28, 28, 1))
plot_model(model, to_file='model.png')
print(model.summary())


def model_vision():
    # First, define the vision modules
    digit_input = Input(shape=(28, 28, 1))
    x = Conv2D(64, (3, 3))(digit_input)
    x = Conv2D(64, (3, 3))(x)
    x = MaxPooling2D((2, 2))(x)
    out = Flatten()(x)

    vision_model = Model(digit_input, out)

    # Then define the tell-digits-apart model
    digit_a = Input(shape=(28, 28, 1))
    digit_b = Input(shape=(28, 28, 1))

    # The vision model will be shared, weights and all
    out_a = vision_model(digit_a)
    out_b = vision_model(digit_b)

    concatenated = concatenate([out_a, out_b])
    out = Dense(1, activation='sigmoid')(concatenated)

    classification_model = Model([digit_a, digit_b], out)

    return classification_model


def new_model():

    inputs = Input(shape=(28, 28, 1))

    x = Cropping2D(cropping=((1, 1), (1, 1)))(inputs)
    x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


model = new_model()
print(model.summary())


