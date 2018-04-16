"""
 date: 2018-04-09
 centerloss for facescurb datasets
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dropout, BatchNormalization, Dense, PReLU
from keras.layers import Embedding, Multiply, Lambda
from keras.models import Model, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import keras.backend as K
import numpy as np


K.set_image_data_format('channels_last')
train_path = '/Users/jmc/Desktop/facepaper/face_data/face_tra_aug/'
val_path = '/Users/jmc/Desktop/facepaper/face_data/face_val/'
BATCH_SIZE = 64
SIZE = (224, 224)
EPOCHS = 20
OUT_DIMS = 529
mean_image = 121.58
std_image = 67.6
Feats_Dims = 1024
sum_trian = 123520
sum_val = 7199

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


# one-hot转换成普通label
def translate_onehot2label(one_hot):

    # length = num of images labels, nb_classes = classes of image
    length = one_hot.shape[0]
    nb_classes = one_hot.shape[1]

    labels = []
    for i in range(length):
        for j in range(nb_classes):
            if one_hot[i][j] == 1:
                labels.append(j)

    labels = np.array(labels).reshape((length, 1))

    return labels


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
        # 非one-hot编码
        y_value = translate_onehot2label(y)

        random_centers = np.random.randn(BATCH_SIZE, 1)

        data_x = [x, y_value]
        data_y = [y, random_centers]
        yield data_x, data_y


def my_val_generator(generator):
    """
    :param generator:
    :return: x: [x, y_value], y: [y, random_centers]
    """
    n = 1
    while True:

        data = next(generator)
        x, y = data[0], data[1]
        # 图片标准化处理
        x = (x - mean_image) / std_image
        # 非one-hot编码
        y_value = translate_onehot2label(y)

        if n <= (7119 // BATCH_SIZE):
            random_centers = np.random.randn(BATCH_SIZE, 1)

            data_x = [x, y_value]
            data_y = [y, random_centers]

            yield data_x, data_y

        else:
            random_centers = np.random.randn(15, 1)

            data_x = [x, y_value]
            data_y = [y, random_centers]

            yield data_x, data_y
        n = n+1


def build_model(isCenterloss, lambda_c):
    if isCenterloss:
        resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

        for layer in resnet_base.layers:
            layer.trainable = True

        # add flatten and features vector
        x = resnet_base.get_layer('avg_pool').output
        x = Flatten(name='flatten')(x)
        x = Dense(Feats_Dims, activation='relu', name='feats_dense')(x)
        fc = Dropout(0.5, name='droput_1')(x)
        x = BatchNormalization(name='bn_dense_1')(x)
        x = Dense(OUT_DIMS, activation='softmax', name='softmax_dense')(x)

        lambda_c = lambda_c  # this parama is variable (total_loss = softmax + lambda_c*l2_loss)
        input_target = Input(shape=(1,))  # single value ground truth labels as inputs
        centers = Embedding(OUT_DIMS, Feats_Dims)(input_target)  # 类中心映射
        l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([fc, centers])
        model_centerloss = Model(inputs=[resnet_base.input, input_target], outputs=[x, l2_loss])
        model_centerloss.compile(optimizer='sgd', loss=['categorical_crossentropy', lambda y_true, y_pred: y_pred],
                                 loss_weights=[1., lambda_c], metrics=['accuracy'])
        return model_centerloss

    else:
        resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

        for layer in resnet_base.layers:
            layer.trainable = True

        # add flatten and features vector
        x = resnet_base.get_layer('avg_pool').output
        x = Flatten(name='flatten')(x)
        x = Dense(Feats_Dims, activation='relu', name='feats_dense')(x)
        fc = Dropout(0.5, name='droput_1')(x)
        x = BatchNormalization(name='bn_dense_1')(fc)
        x = Dense(OUT_DIMS, activation='softmax', name='softmax_dense')(x)

        # build model
        resnet_model = Model(inputs=resnet_base.input, outputs=x)
        return resnet_model


def lrschedule(epoch):
    if epoch <= 10:
        return 0.1
    elif epoch <= 20:
        return 0.05
    elif epoch <= 30:
        return 0.01
    else:
        return 0.005


model = build_model(True, lambda_c=0.01)
train_generator_mygenerator = mygenerator(train_generator)
test_generator_mygenerator = mygenerator(test_generator)

data_train = next(train_generator_mygenerator)
print(data_train[0][0].shape,
      data_train[0][1].shape,
      data_train[1][0].shape,
      data_train[1][1].shape)


weights_path = 'best_model.h5'
modelcheckpoint = ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True, mode='auto')
lr = LearningRateScheduler(lrschedule)
history_centerloss = model.fit_generator(train_generator_mygenerator, steps_per_epoch=sum_trian // BATCH_SIZE,
                                         epochs=EPOCHS,
                                         validation_data=test_generator_mygenerator,
                                         validation_steps=sum_val // BATCH_SIZE,
                                         callbacks=[lr, modelcheckpoint])



