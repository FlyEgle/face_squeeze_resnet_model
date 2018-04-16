from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dropout, BatchNormalization, Dense, PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
import keras.backend as K
import os

Feats_Dims = 1024
OUT_DIMS = 529
resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# not freeze imageweights in resnet_base layer
for layer in resnet_base.layers:
    layer.trainable = True

# add flatten and features vector
x = resnet_base.get_layer('avg_pool').output
x = Flatten(name='flatten')(x)
x = Dropout(0.5, name='droput_1')(x)
x = BatchNormalization(name='bn_1')(x)
x = Dense(Feats_Dims, activation='relu', name='feats_dense')(x)
x = Dropout(0.5, name='droput_2')(x)
x = BatchNormalization(name='bn_dense_1')(x)
x = Dense(OUT_DIMS, activation='softmax', name='softmax_dense')(x)

# build model
resnet_model = Model(inputs=resnet_base.input, outputs=x)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
K.set_image_data_format('channels_last')
train_path = '/Users/jmc/Desktop/facepaper/face_data/face_tra_aug/'
val_path = '/Users/jmc/Desktop/facepaper/face_data/face_val/'
BATCH_SIZE = 64
EPOCHS = 40
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
    elif epoch <=20:
        return 0.05
    elif epoch <=30:
        return 0.01
    else:
        return 0.005

lr = LearningRateScheduler(lrschedule)
model_weights_path = 'best_epoch.h5'
modelcheckpoint = ModelCheckpoint(model_weights_path, monitor='val_acc',
                                 save_best_only=True, mode='auto')
# train data generator
train_generator = generator_from_directory(data_train_agu(), train_path, SIZE, BATCH_SIZE, 'categorical')
# val data generator
val_generator = generator_from_directory(data_val_agu(), val_path, SIZE, BATCH_SIZE, 'categorical')
# resnet_model.load_weights(model_weights_path)
resnet_model.compile(optimizer=SGD(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

history_softmax = resnet_model.fit_generator(
        train_generator,
        steps_per_epoch=123520//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=7119//BATCH_SIZE,
        callbacks=[lr, modelcheckpoint]
    )

