# -*- coding: utf-8 -*-
"""
    把数据集文件名字重新命名成1~len(folder)
"""
import os


def get_foldername(path):

    old_foldername = []
    old_path = []
    for folder in os.listdir(path):
        if not folder == '.DS_Store':
            old_foldername.append(folder)
            old_path.append(path+folder)
    return old_foldername, old_path


def newname(path):

    old_foldername, old_path = get_foldername(path)
    new_foldername = [str(x+1) for x in range(len(old_foldername))]
    return new_foldername


def change_folder_name(path):

    old_foldername, old_path = get_foldername(path)
    new_path = newname(path)
    new_foldername = []
    for folder in new_path:
        new_foldername.append(path+folder)

    for i in range(len(new_foldername)):
        os.rename(old_path[i], new_foldername[i])



# if __name__ == '__main__':
#
#     path = '/Users/jmc/Desktop/CASIA-WebFace-FirstClean/'
#     change_folder_name(path)


