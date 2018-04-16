from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, BatchNormalization
from keras.models import Model

model = MobileNet(input_shape=(224, 224, 3),
                  include_top=True)


x = model.get_layer('global_average_pooling2d_1').output
x = BatchNormalization()(x)
x = Dense(529, activation='softmax')(x)

newmodel = Model(inputs=model.input, outputs=x)
print(newmodel.summary())


