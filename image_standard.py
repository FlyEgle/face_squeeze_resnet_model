from keras.applications.inception_resnet_v2 import InceptionResNetV2

model = InceptionResNetV2(include_top=False, weights='imagenet')

