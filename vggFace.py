from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Conv2D
from PIL import Image
from keras.optimizers import SGD
import numpy as np
import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array
K.set_image_data_format('channels_first')


def vgg_face(weights_path=True):
    img = Input(shape=(3, 224, 224))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

    flat = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flat)
    fc6_drop = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu', name='fc7')(fc6_drop)
    fc7_drop = Dropout(0.5)(fc7)
    out = Dense(2622, activation='softmax', name='fc8')(fc7_drop)

    model = Model(inputs=img, outputs=out)

    if weights_path:
        model.load_weights(weights_path)

    return model


def model_extraction(weight_path):

    base_model = vgg_face(weight_path)
    model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('fc7').output)

    return model


if __name__ == "__main__":
    weight_path = '/Users/jmc/Desktop/facepaper/vggface/vgg-face-keras-fc.h5'
    im = Image.open('/Users/jmc/Desktop/facepaper/lfw_2/Abel_Pacheco/Abel_Pacheco_0001.jpg')
    im = im.resize((224, 224))
    im = np.array(im).astype(np.float32)
    # im[:,:,0] -= 129.1863
    # im[:,:,1] -= 104.7624
    # im[:,:,2] -= 93.5940
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    base_model = vgg_face(weight_path)
    print(base_model.summary())

    model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('fc7').output)

    print(model.summary())

    img_feats = model.predict(im)

    print(img_feats.shape)
    # print(img_feats[0][0].shape)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_feats[0][3])
    # plt.show()
    print(sum(img_feats[0]))


