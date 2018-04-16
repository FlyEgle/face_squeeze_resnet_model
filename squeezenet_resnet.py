from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from train_model import squeezenet
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.applications.mobilenet import MobileNet
import keras.backend as K

BATCH_SIZE = 64
EPOCHS = 50
SIZE = (224, 224)
OUT_DIMS = 529  # face scrub class
train_path = '/Users/jmc/Desktop/facepaper/face_data/face_tra_aug/'
val_path = '/Users/jmc/Desktop/facepaper/face_data/face_val/'
K.set_image_data_format('channels_last')
model = MobileNet(input_shape=(224, 224, 3), include_top=False, classes=529)
# x = base_model.get_layer('relu_conv10').output
# x = BatchNormalization()(x)
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.7)(x)
# x = Dense(529, activation='softmax')(x)
print(model.summary())
# model = Model(inputs=base_model.input, outputs=x)


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


# learningRate
def lrschedule(epoch):

    if epoch <= 10:
        return 0.1
    elif epoch <= 20:
        return 0.005
    elif epoch <= 30:
        return 0.001
    elif epoch <= 40:
        return 0.0005
    else:
        return 0.0001


# modelcheckpoint
model_check = ModelCheckpoint(filepath='weights_best_squeezenet.h5', monitor='val_acc', save_best_only=True)
lr = LearningRateScheduler(lrschedule)


def train_model():
    # train data generator
    train_generator = generator_from_directory(data_train_agu(), train_path, SIZE, BATCH_SIZE, 'categorical')
    # val data generator
    val_generator = generator_from_directory(data_val_agu(), val_path, SIZE, BATCH_SIZE, 'categorical')
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # fit_generator
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=123520 // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=7119 // BATCH_SIZE,
        callbacks=[lr, model_check]
    )


train_model()

