"""
#-*- coding = utf-8 -*- 
#@Time: 2022 11 04  下午6:47
#@Author:JFZ
#@File：utils.py
#@Software: PyCharm
"""
import random

import cairosvg
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil

# ===============================文件夹处理==================================
def check_dictionary(path):
    """不存在文件夹则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


# ==========================图像处理=======================================
def svg2png(svg_path="/home/harry/Python_Demo/Image_Restoration/PaddleBased/HIWR/version1/datasets/Adidas.svg", save_path="1.png", h=None, w=None, ):
    """SVG转PNG"""
    if h == None or w == None:
        cairosvg.svg2png(url=svg_path, write_to=save_path, dpi=96, )
    cairosvg.svg2png(url=svg_path, write_to=save_path, dpi=96, output_width=w, output_height=h)


def concat_image(image_list, axis=2):
    """将ndarray的image在某个通道进行拼接，传入的图像要保证是一个列表或者元组"""
    return np.concatenate(image_list, axis=axis)


def ToNumpy(*tensors):
    """批量将图像转为numpy形式，
    inputs : list or tuple or iterable, shape = [C,H,W]
    :return: list
    """
    out = list()
    for tensor in tensors:
        img = tensor.transpose([1, 2, 0]).cpu().numpy()
        # img = Image.fromarray(img)
        out.append(img)
    return out


# =============================可视化处理=====================================
def showImage(image, title="Image",cmap=None):
    """传入Ndarray形式 or"""
    if cmap is None:
        plt.imshow(image, )
    else:
        plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def showImages(*images, titles=None):
    concat_image = []
    for image in images:
        assert len(image.shape) == 3, "应该保障图像遵守HWC这三个维度"
        if image.shape[-1] == 1:  # 单通道图像，则将图像进行堆叠
            image = np.concatenate([image for i in range(3)], axis=-1)
        concat_image.append(image)
    concat_image = np.concatenate([image for image in concat_image], axis=1)
    showImage(concat_image)


# =====================数据增强====================

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
    # if random.random() > 0:
        mode = random.randint(0, 7)
        for image in images:
            image = data_augmentation(image=image, mode=mode)
            out.append(image)
        return out
    else:
        return images
