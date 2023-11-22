# coding=utf-8
"""
    @Project: RWRNet_Paddle
    @Author：JFZ
    @File： SDIWRD.py
    @Date：2023/11/21 17:05
    @EnvConfig: pytorch 1.12.1 + cu116
"""
from paddle.io import Dataset
from paddle.vision.transforms import ToTensor
from datasets import random_augmentation
import numpy as np
from PIL import Image
import os
import os.path as osp


class LoadSDIWRD(Dataset):
    def __init__(self, root=r"E:\Python_Demo\Datasets\SDIWRD", train=True):
        super(LoadSDIWRD, self).__init__()
        self.root = root
        self.type = "train" if train else "test"

        self.wm = osp.join(root, self.type, "images")
        self.gt = osp.join(root, self.type, "gts")

        if not train:
            self.IDs = [str(i) for i in range(1, 1314 + 1)]
        else:
            self.IDs = [file_path.strip(".jpg") for file_path in os.listdir(self.wm)]

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

# if __name__ == '__main__':
#     from paddle.io import DataLoader
#     dataset = LoadSDIWRD()
#     dataset = DataLoader(dataset,batch_size=1)
#     print(len(dataset))
#     print(next(iter(dataset)))
