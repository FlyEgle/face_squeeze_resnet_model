from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dropout, BatchNormalization, Dense, PReLU
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.losses import categorical_crossentropy
import keras.backend as K
import os
import numpy as np

# default params for all model and dataset path change
os.environ['CUDA_VISIBLE_DEVICE'] = "1"
K.set_image_data_format('channels_last')
train_path = '/media/wislab/DataSet/jiang/FaceDataSet/face_data/face_tra_aug/'
val_path = '/media/wislab/DataSet/jiang/FaceDataSet/face_data/face_val/'
BATCH_SIZE = 128
EPOCHS = 20
SIZE = (224, 224)
Feats_Dims = 1024
OUT_DIMS = 529
model_weights_path = 'best_epoch.hdf5'


# build cnn pre_train model
def build_model():

    resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # freeze imageweights in resnet_base layer
    for layer in resnet_base.layers:
        layer.trainable = False

    # add flatten and features vector
    x = resnet_base.get_layer('avg_pool').output
    x = Flatten(name='flatten')(x)
    x = Dense(Feats_Dims, name='feats_dense')(x)
    x = PReLU(name='prelu_1')(x)
    x = BatchNormalization(name='bn_dense_1')(x)
    x = Dropout(0.5, name='droput_1')(x)
    x = Dense(OUT_DIMS, activation='softmax', name='softmax_dense')(x)

    # build model
    resnet_model = Model(inputs=resnet_base.input, outputs=x)
    return resnet_model


# class of data precessing
class DataPrepare(object):

    @staticmethod
    def data_train_agu():
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            rescale=1./255
        )
        return datagen

    @staticmethod
    def data_val_agu():
        datagen = ImageDataGenerator(
            rescale=1./255
        )
        return datagen

    @staticmethod
    def generator_from_directory(datagen, path, target_size, batch_size, classmode):
        generator = datagen.flow_from_directory(
            path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=classmode
        )
        return generator

    @staticmethod
    def lrschedule(epoch):
        if epoch <= 10:
            return 0.1
        else:
            return 0.01

    @staticmethod
    def learningreate():
        return LearningRateScheduler(DataPrepare.lrschedule)

    @staticmethod
    def modelcheckpoint():
        modelcheck = ModelCheckpoint(model_weights_path, monitor='val_acc', save_best_only=True,
                                     mode='auto')
        return modelcheck

    @staticmethod
    def earlystopping():
        earlystop = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
        return earlystop

    @staticmethod
    def reducelronplateau():
        reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')
        return reducelr


# train,val data generator
def data_generator():
    datapre = DataPrepare()
    train_generator = datapre.generator_from_directory(datapre.data_train_agu(),
                                                       train_path,
                                                       SIZE,
                                                       BATCH_SIZE,
                                                       'categorical')
    val_generator = datapre.generator_from_directory(datapre.data_val_agu(),
                                                     val_path,
                                                     SIZE,
                                                     BATCH_SIZE,
                                                     'categorical')
    return train_generator, val_generator


def model_train(model, optimizers, losses):

    datapre = DataPrepare()
    lr = datapre.learningreate()
    modelcheckpoint = datapre.modelcheckpoint()
    earlystop = datapre.earlystopping()
    reducelr = datapre.reducelronplateau()

    callbacks = [lr, modelcheckpoint, earlystop, reducelr]
    train_generator, val_generator = data_generator()


    # model.compile(optimizer=optimizers, loss=losses, metrics=['accuracy'])
    # history = model.fit_generator(train_generator, steps_per_epoch=12350//BATCH_SIZE, epochs=EPOCHS,
    #                               validation_data=val_generator, validation_steps=7119//BATCH_SIZE, callbacks=callbacks)
    #
    # return history


def main():

    model = build_model()
    # history = model_train(model, optimizers=SGD(momentum=0.9, nesterov=True), losses=categorical_crossentropy)
    #
    # np.save('history.npy', history)


if __name__ == '__main__':
    main()










