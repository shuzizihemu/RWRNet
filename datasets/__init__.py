# coding=utf-8
"""
    @Project: RWRNet_Paddle
    @Author：JFZ
    @File： __init__.py.py
    @Date：2023/11/21 16:57
    @EnvConfig: pytorch 1.12.1 + cu116
"""
import os
import numpy as np
import random


def data_augmentation(image, mode=0):
    """
    对输入图像进行数据增强
    :param image:
    :param mode:
    :return:
    """
    if mode == 0:  # 什么也不做
        return image
    elif mode == 1:  # 上下镜像
        return np.flipud(image)
    elif mode == 2:  # 顺时针旋转90度
        return np.rot90(image)
    elif mode == 3:  # 顺时针旋转90，然后上下镜像
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        return np.rot90(image, k=2)  # rotate 180 degree
    elif mode == 5:
        image = np.rot90(image, k=2)  # rotate 180 degree and flip
        return np.flipud(image)
    elif mode == 6:
        return np.rot90(image, k=3)  # rotate 270 degree
    elif mode == 7:
        image = np.rot90(image, k=3)  # rotate 270 degree and flip
        return np.flipud(image)
    else:
        raise Exception('Invalid choice of image transformation')


def random_augmentation(*images):
    """对图像进行随机数据增强,若送来的是一批图像，那么这一批图像进行相同的数据增强"""
    out = list()
    if random.random() > 0.5:
        mode = random.randint(0, 7)
        for image in images:
            image = data_augmentation(image=image, mode=mode)
            out.append(image)
        return out
    else:
        return images
