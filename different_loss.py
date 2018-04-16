from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop, SGD, Adadelta
from keras.metrics import categorical_accuracy
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


K.set_image_data_format('channels_last')
batch_size = 128
num_classes = 10
epochs = 50
learning_rate = 0.1

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'train sample')
print(x_test.shape[0], 'test sample')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
tensorboard = TensorBoard(log_dir='./logs', batch_size=batch_size)


# definite callback
class MyCallBack(Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        self.auccarcy = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        print('\n======')
        print(self.validation_data[0].shape)
        print(self.validation_data[1].shape)
        print('========')

        input = self.model.input
        labels = np.argmax(self.validation_data[1], axis=1)
        layers_model = Model(inputs=input, outputs=self.model.get_layer('feats').output)
        fc_output = layers_model.predict(self.validation_data[0])

        visualize_feats(fc_output, labels, epoch)

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_train_end(self, logs=None):
        return


def visualize_feats(feat, labels, epoch):

    label_vector = 10
    plt.ion()
    c = ['#000000', '#0066FF', '#FF0000', '#FF9900', '#FFCC00',
         '#00CC00', '#FF6600', '#00FF00', '#CC0066', '#003333']
    plt.clf()
    plt.figure(figsize=(10, 6))
    for i in range(label_vector):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], ',', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='best')
    XMax = np.max(feat[:, 0])
    YMax = np.max(feat[:, 1])
    XMin = np.min(feat[:, 0])
    YMin = np.min(feat[:, 1])

    plt.xlim(xmin=XMin, xmax=XMax)
    plt.ylim(ymin=YMin, ymax=YMax)
    plt.text(XMin, YMax, 'epoch=%d' % epoch)
    plt.draw()
    plt.pause(0.001)


# triplet_loss
def triplet_loss(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred, axis=1)
    batch = batch_size
    ref1 = y_pred[0:batch:1]
    pos1 = y_pred[batch:batch+batch, :]
    neg1 = y_pred[batch:batch:3*batch, :]
    dis_pos = K.sum(K.square(ref1-pos1), axis=1, keepdims=True)
    dis_neg = K.sum(K.square(ref1-neg1), axis=1, keepdims=True)
    dis_pos = K.sqrt(dis_pos)
    dis_neg = K.sqrt(dis_neg)
    a1 = 17
    d1 = dis_pos+K.maximum(0.0, dis_pos - dis_neg + a1)
    return K.mean(d1)

alpha = 0.01
# l2-softmax
def l2_softmax(y_true, y_pred):
    return alpha*K.l2_normalize(K.categorical_crossentropy(y_true, y_pred), axis=1)



def builde_model(input):
    input_shape = Input(input)
    x = Conv2D(32, (3, 3), activation='relu')(input_shape)
    x = Flatten()(x)
    x = Dense(64, activation='relu', name='feats')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(input_shape, x)
    return model


mycallback = MyCallBack()
model = builde_model((28, 28, 1))
model.compile(loss=triplet_loss, optimizer=Adadelta(lr=learning_rate), metrics=[categorical_accuracy])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                    callbacks=[tensorboard])

score = model.evaluate(x_test, y_test)
print('loss: ', score[0])
print('acc: ', score[1])

x_trick = [x+1 for x in range(epochs)]
loss = history.history['loss']
acc = history.history['categorical_accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_categorical_accuracy']

plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.title('learninngRate = %f, batch_size = %s' % (0.1, batch_size))
plt.plot(x_trick, loss, 'g-', label='loss')
plt.plot(x_trick, val_loss, 'r-', label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

plt.figure(figsize=(10, 6))
plt.title('learninngRate = %f, batch_size = %s' % (0.1, batch_size))
plt.plot(x_trick, val_acc, 'y-', label='val_acc')
plt.plot(x_trick, acc, 'b-', label='acc')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()

