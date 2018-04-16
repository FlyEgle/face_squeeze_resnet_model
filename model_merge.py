from keras import Input
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.layers import concatenate
from keras.layers import Dropout, Dense, maximum

input_shape = Input(shape=(224, 224, 3))
base_model1 = Xception(include_top=True, weights='imagenet')
base_model1 = Model(inputs=[base_model1.input], outputs=[base_model1.get_layer('avg_pool').output],
                    name='xception')

base_model2 = InceptionV3(include_top=True, weights='imagenet')
base_model2 = Model(inputs=[base_model2.input], outputs=[base_model2.get_layer('avg_pool').output],
                    name='inceptionv3')

img1 = Input(shape=(224, 224, 3), name='img_1')

feature1 = base_model1(img1)
feature2 = base_model2(img1)

categorical_predic1 = Dense(100, activation='softmax', name='ctg_out_1')(
    Dropout(0.5)(feature1)
)

categorical_predic2 = Dense(100, activation='softmax', name='ctg_out_2')(
    Dropout(0.5)(feature2)
)

categorcial_predict = Dense(100, activation='softmax', name='ctg_out')(
    concatenate([feature1, feature2])
)

max_category_predict = maximum([categorical_predic1, categorical_predic2])

model = Model(inputs=[img1], outputs=[categorical_predic1, categorical_predic2, categorcial_predict, max_category_predict])

print(model.summary())
