# -*- coding:utf-8 -*-
"""
image arguement for image
random_crop
random_flip_left_right
random_flip_left_right
random_contrast
random_brightness
random_color
per_image_standardization
"""
from PIL import Image
from PIL import ImageEnhance
from random import randint
from skimage.util import random_noise
import random
import numpy as np

# img = Image.open('1.jpg')
# print img.format, img.size, img.mode
# img.resize((1080,768))
# img.crop((14,14,79,79)).show()
# print img.getpixel((1920,1080))


def random_crop(img, width, height):
    width1 = randint(0, img.size[0] - width )
    height1 = randint(0, img.size[1] - height)
    width2 = width1 + width
    height2 = height1 + height
    img = img.crop((width1, height1, width2, height2))
    return img


def random_flip_left_right(img):
    prob = randint(0,1)
    if prob == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_contrast(img, lower=0.2, upper=1.8):
    factor = random.uniform(lower, upper)
    img = ImageEnhance.Sharpness(img)
    img = img.enhance(factor)
    return img


def random_brightness(img, lower=0.6, upper=1.4):
    factor = random.uniform(lower, upper)
    img = ImageEnhance.Brightness(img)
    img = img.enhance(factor)
    return img


def random_color(img, lower=0.6, upper=1.5):
    factor = random.uniform(lower, upper)
    img = ImageEnhance.Color(img)
    img = img.enhance(factor)
    return img


def random_gaussiannoise(img, mode='gaussian'):
    """
    add gaussian noise
    """
    noise_img = random_noise(img)

    return noise_img


def per_image_standardization(img):
    '''
    stat = ImageStat.Stat(img)
    mean = stat.mean
    stddev = stat.stddev
    img = (np.array(img) - stat.mean)/stat.stddev
    '''
    global channel
    if img.mode == 'RGB':
        channel = 3
    num_compare = img.size[0] * img.size[1] * channel
    img_arr=np.array(img)
    #img_arr=np.flip(img_arr,2)
    img_t = (img_arr - np.mean(img_arr))/max(np.std(img_arr), 1/num_compare)
    return img_t

#img = random_crop(img,1000,1000)
#img = random_flip_left_right(img)
'''img = ImageEnhance.Sharpness(img)
img.enhance(5.0).show()'''
#img = random_contrast(img, 4,6)
