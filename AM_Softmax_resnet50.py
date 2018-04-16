"""
amsoftmax for facescrub
"""
from Am_softmax import AMSoftmax, amsoftmax_loss
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dropout, BatchNormalization, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import keras.backend as K
import os


Feats_Dims = 1024
OUT_DIMS = 529
BATCH_SIZE = 64
EPOCHS = 40
SIZE = (224, 224)
mean_image = 121.58
std_image = 67.6

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
K.set_image_data_format('channels_last')
train_path = '/Users/jmc/Desktop/facepaper/face_data/face_tra_aug/'
val_path = '/Users/jmc/Desktop/facepaper/face_data/face_val/'

resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

for layer in resnet_base.layers:
    layer.trainable = True

x = resnet_base.get_layer('avg_pool').output
x = Flatten(name='flatten')(x)
x = Dropout(0.7, name='droput_1')(x)
x = BatchNormalization(name='bn_1')(x)
x = Dense(Feats_Dims, activation='relu', name='feats_dense')(x)
x = Dropout(0.3, name='droput_2')(x)
x = BatchNormalization(name='bn_dense_1')(x)

# am softmax dense, m is the margin of cos (theta) is a variable params
x = AMSoftmax(OUT_DIMS, OUT_DIMS, m=0.1)(x)

resnet_model = Model(inputs=resnet_base.input, outputs=x)


# 图像增强和生成器
train_datagen = ImageDataGenerator(
    horizontal_flip=True
)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    val_path,
    target_size=SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


def lrschedule(epoch):
    if epoch <= 10:
        return 0.1
    elif epoch <= 20:
        return 0.05
    elif epoch <= 30:
        return 0.01
    else:
        return 0.005


# 自己的生成器
def mygenerator(generator):
    """
    :param generator:
    :return: x: [x, y_value], y: [y, random_centers]
    """

    while True:
        data = next(generator)
        x, y = data[0], data[1]
        # 图片标准化处理
        x = (x - mean_image) / std_image
        data_x, data_y = x, y
        yield data_x, data_y


train_generator_mygenerator = mygenerator(train_generator)
val_generator_mygenerator = mygenerator(test_generator)

weights_path = 'best_amsoftmax_0_1.h5'
modelcheckpoint = ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True, mode='auto')
lr = LearningRateScheduler(lrschedule)

resnet_model.compile(loss=amsoftmax_loss, optimizer=Adam(), metrics=['accuracy'])
history_amsoftmax = resnet_model.fit_generator(train_generator_mygenerator,
                                               steps_per_epoch=123520 // BATCH_SIZE,
                                               epochs=EPOCHS,
                                               validation_data=val_generator_mygenerator,
                                               validation_steps=7040 // BATCH_SIZE,
                                               callbacks=[lr, modelcheckpoint])

