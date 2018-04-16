# -*- coding: utf-8 -*-
"""
1 proprecessing data to trainning and validation
2 proprecessing data to arguement
    random_crop 2
    random_gaussian randint
当前用的数据集directory是face_data

"""
from PIL import Image
from imageaug import random_crop, random_gaussiannoise
import os
import matplotlib.pyplot as plt
import numpy as np


def compute_count(src):
    global count
    count_list = []
    for folder in os.listdir(src):
        if not folder == '.DS_Store':
            count = 0
            for image in os.listdir(src+folder):
                if not image == '.DS_Store':
                    count += 1
        count_list.append(count)

    return count_list


def show_data(count):
    x = [_ for _ in range(len(count))]
    plt.figure(figsize=(16, 6))
    plt.bar(x, count)
    plt.show()


def imgaug_resize(src, dst, times=2, width=224, height=224):
    """
    :param src: face image of source directory
    :param dst: face image of destination directory
    :param times: times for random crop
    :param width: crop width
    :param height: crop height
    :return: new image folder
    """
    for folder in os.listdir(src):
        if not folder == '.DS_Store':
            new_folder = os.path.join(dst, folder)
            os.mkdir(new_folder)
            image_path = os.path.join(src, folder)
            for image in os.listdir(image_path):
                if not image == '.DS_Store':
                    for _ in range(times):
                       img= Image.open(os.path.join(image_path, image))
                       crop_img = random_crop(img, width, height)
                       crop_img.save(new_folder+'/'+str(image.split('.jpg')[0])+'random_crop'+str(_)+'.jpg')


def imgaug_random_noise(image):
    """
    图片添加噪声
    :param image: PIL class image
    :return: PIL write image
    """
    arr_image = np.array(image)
    noise_image = random_gaussiannoise(arr_image)
    image = Image.fromarray(noise_image)
    return image


if __name__ == '__main__':

    src_path = '/Users/jmc/Desktop/facepaper/face_data/face_tra/'
    dst_path = '/Users/jmc/Desktop/facepaper/face_data/face_tra_aug/'
    data_path = '/Users/jmc/Desktop/facepaper/face_data/'

    # cout = compute_count(dst_path)
    # show_data(cout)
    print("=========")
    print("start!")
    imgaug_resize(src_path, dst_path)
    print("end!")
