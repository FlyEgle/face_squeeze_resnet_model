from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from train_model import squeezenet
from keras_squeezenet import SqueezeNet
from keras.losses import categorical_crossentropy
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.utils import multi_gpu_model
import keras.backend as K
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
K.set_image_data_format('channels_last')
train_path = '/Users/jmc/Desktop/facepaper/face_data/face_tra_aug/'
val_path = '/Users/jmc/Desktop/facepaper/face_data/face_val/'
BATCH_SIZE = 64
EPOCHS = 10
SIZE = (224, 224)
OUT_DIMS = 529  # face scrub class


def data_train_agu():
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rescale=1./255
    )
    return datagen


def data_val_agu():
    datagen = ImageDataGenerator(
        rescale=1./255
    )
    return datagen


# train, val generator from directory
def generator_from_directory(datagen, path, target_size, batch_size, classmode):
    generator = datagen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=classmode
    )
    return generator


def lrschedule(epoch):
    if epoch <= 10:
        return 0.1
    elif epoch <= 20:
        return 0.05
    elif epoch <=30:
        return 0.001
    elif epoch <=40:
        return 0.005
    else:
        return 0.0001


# 学习率
lr_schedule = LearningRateScheduler(lrschedule)
# train data generator
train_generator = generator_from_directory(data_train_agu(), train_path, SIZE, BATCH_SIZE, 'categorical')
# val data generator
val_generator = generator_from_directory(data_val_agu(), val_path, SIZE, BATCH_SIZE, 'categorical')
squeezenet_model = squeezenet(input_shape=(224, 224, 3), out_dims=OUT_DIMS)

print(squeezenet_model.get_layer('dense_1').output)

print(squeezenet_model.summary())
squeezenet_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
squeezenet_model.fit_generator(
        train_generator,
        steps_per_epoch=123520//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=7119//BATCH_SIZE
)
