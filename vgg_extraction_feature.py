#coding=utf-8
#keras==0.3.0 theano==0.8.0 python==2.7.13
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np, time

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def VGG_16_bymodel(model):
    model2 = Sequential()
    model2.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model2.add(Convolution2D(64, 3, 3, activation='relu', weights=model.layers[1].get_weights()))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(64, 3, 3, activation='relu', weights=model.layers[3].get_weights()))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))

    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(128, 3, 3, activation='relu', weights=model.layers[6].get_weights()))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(128, 3, 3, activation='relu', weights=model.layers[8].get_weights()))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))

    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(256, 3, 3, activation='relu', weights=model.layers[11].get_weights()))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(256, 3, 3, activation='relu', weights=model.layers[13].get_weights()))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(256, 3, 3, activation='relu', weights=model.layers[15].get_weights()))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))

    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[18].get_weights()))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[20].get_weights()))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[22].get_weights()))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))

    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[25].get_weights()))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[27].get_weights()))
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Convolution2D(512, 3, 3, activation='relu', weights=model.layers[29].get_weights()))
    model2.add(MaxPooling2D((2,2), strides=(2,2)))

    model2.add(Flatten())
    model2.add(Dense(4096, activation='relu', weights=model.layers[32].get_weights()))
    model2.add(Dropout(0.5))
    model2.add(Dense(4096, activation='relu', weights=model.layers[34].get_weights()))
    # model2.add(Dropout(0.5))
    # model2.add(Dense(1000, activation='softmax'))

    return model2

#分类图像
# im = cv2.resize(cv2.imread('zebra.jpg'),(224,224)).astype(np.float32)
# im[:,:,0] -= 103.939
# im[:,:,1] -= 116.779
# im[:,:,2] -= 123.68
# im = im.transpose((2,0,1))
# im = np.expand_dims(im, axis=0)
#
# # Test pretrained model
# start_time=time.time()
# model = VGG_16('vgg16_weights.h5')
# print('load model time:',time.time()-start_time)
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')
# start_time=time.time()
# out = model.predict(im)
# print('predict time:',time.time()-start_time)
# print np.argmax(out),out[0][np.argmax(out)]
# f = open('synset_words.txt','r')
# lines = f.readlines()
# f.close()
# print(lines[np.argmax(out)])

#提取特征
model = VGG_16('vgg16_weights.h5')
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')
print(len(model.layers))
layer_count=len(model.layers)
for i,layer in enumerate(model.layers):
    print('layer:',i+1)
    print(len(layer.get_weights()))

im = cv2.resize(cv2.imread('/Users/jmc/Desktop/zebra.jpg'),(224,224)).astype(np.float32)
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
model2=VGG_16_bymodel(model)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(optimizer=sgd, loss='categorical_crossentropy')
out2=model2.predict(im)
print(len(out2[0]), out2)
