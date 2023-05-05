"""
#-*- coding = utf-8 -*- 
#@Time: 2022 11 09  下午6:45
#@Author:JFZ
#@File：SDIWRD.py
#@Software: PyCharm
    加载数据
"""

import os
import os.path as osp
from paddle.io import Dataset
from paddle.vision.transforms import ToTensor
from datasets.utils import random_augmentation
import numpy as np
from PIL import Image


class LoadSDIWRD(Dataset):
    def __init__(self, root="/home/harry/Python_Demo/Datasets/SDIWRD", train=True):
        super(LoadSDIWRD, self).__init__()
        self.root = root
        self.type = "train" if train else "test"
        # 数据根目录
        self.wm = osp.join(root, self.type, "images")  # watermark's paths. exp: "/home/harry/Python_Demo/Datasets/CWD/train/wms"
        self.gt = osp.join(root, self.type, "gts")
        # self.mask = osp.join(root, self.type, "masks")

        # 数据ID列表
        if not train:
            self.IDs = [str(i) for i in range(1, 1314 + 1)]
        else:
            self.IDs = [file_path.strip(".jpg") for file_path in os.listdir(self.wm)]  # ["0", "1, "2", "3", ...]
        # 数据增强
        self.totensor = ToTensor()
        self.random_aug = random_augmentation

    def __getitem__(self, index):
        img_id = self.IDs[index]
        wm = Image.open(osp.join(self.wm, "{}.jpg".format(img_id))).convert("RGB")
        wm = np.array(wm)
        gt = Image.open(osp.join(self.gt, "{}.jpg".format(img_id))).convert("RGB")
        gt = np.array(gt)

        if self.type == "train":
            wm, gt = random_augmentation(wm, gt)
        gt = self.totensor(gt)
        wm = self.totensor(wm)
        return wm, gt

    def __len__(self):
        return len(self.IDs)
