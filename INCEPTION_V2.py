from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model


def inception_ve_model():
    output_layer = 'conv_7b'
    model_pre_train = InceptionResNetV2(include_top=True,
                                        weights='imagenet',
                                        input_shape=(224, 224, 3))

    model_pre_train.trainable = False
    x = Flatten()(model_pre_train.get_layer(output_layer).output)
    x = Dropout(0.7, name='avg_pool_dropout')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', name='fc_features')(x)
    x = Dropout(0.3, name='dense_dropout')(x)
    x = BatchNormalization()(x)
    predictions = Dense(529, activation='softmax', name='predictions')(x)
    finetuned_model = Model(inputs=model_pre_train.input, outputs=predictions)

    return finetuned_model
