"""
该脚本是用来做data 训练的模板
模型选择从trian_model.py中获取
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Input
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping
from train_model import squeezenet
from keras.losses import categorical_crossentropy
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.utils import multi_gpu_model
from keras_squeezenet.squeezenet import SqueezeNet
import keras.backend as K
import os


inrmodel = InceptionResNetV2(input_shape=(224, 224, 3), weights='imagenet')
print(inrmodel.summary())

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
K.set_image_data_format('channels_last')
train_path = '/Users/jmc/Desktop/facepaper/face_data/face_tra_aug/'
val_path = '/Users/jmc/Desktop/facepaper/face_data/face_val/'
BATCH_SIZE = 32
EPOCHS = 50
SIZE = (224, 224)
OUT_DIMS = 529  # face scrub class

"""
data size = (224, 224)
data arguement:
    水平翻转
    裁剪在本地已经做好
"""


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
        return 0.005
    else:
        return 0.001


# model fune-tune imagenet
def inceptionv3_fine_tune(weights='imagenet', include_top=False, outputdim=OUT_DIMS):

    base_model = InceptionV3(weights=weights, include_top=include_top)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    predictions = Dense(outputdim, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss=categorical_crossentropy, metrics=['accuracy'])
    return model


def train_model():
    # 学习率
    lr_schedule = LearningRateScheduler(lrschedule)
    # train data generator
    train_generator = generator_from_directory(data_train_agu(), train_path, SIZE, BATCH_SIZE, 'categorical')
    # val data generator
    val_generator = generator_from_directory(data_val_agu(), val_path, SIZE, BATCH_SIZE, 'categorical')

    data_train_datagen = data_train_agu()
    # suqeezenetmodel baseline loss is softmax
    # squeezenet_model = squeezenet(input_shape=(224, 224, 3), out_dims=OUT_DIMS)

    # model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet', classes=529)
    # multi GPU 并行
    # parrel_model = multi_gpu_model(model, gpus=2)
    # inceptionvV3_model
    # model = inceptionv3_fine_tune('imagenet', False, OUT_DIMS)
    # # model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    #
    # # fit_generator
    # history = model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=123520//BATCH_SIZE,
    #     epochs=EPOCHS,
    #     validation_data=val_generator,
    #     validation_steps=7119//BATCH_SIZE,
    #     callbacks=[lr_schedule]
    # )
    # model.save('squeeze_softmax.h5')


if __name__ == '__main__':
    train_model()
