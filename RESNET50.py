from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model


def resnet_model():

    Feats_Dims = 1024
    OUT_DIMS = 529
    resnet_base = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))

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

    return resnet_model


