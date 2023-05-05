"""
#-*- coding = utf-8 -*- 
#@Time: 2022 11 18  上午11:11
#@Author:JFZ
#@File：LossFunction.py
#@Software: PyCharm
"""
from paddle import nn
from paddle_ssim import SSIM


class L1Loss(nn.Layer):
    def __init__(self):
        super().__init__()
        self.lossfunc = nn.L1Loss()

    def forward(self, x, y):
        loss = self.lossfunc(x, y)
        return loss


class SSIMLoss(nn.Layer):
    def __init__(self):
        super().__init__()
        self.lossfunc = SSIM()

    def forward(self, pred, gt):
        loss = sum(1 - self.lossfunc(pred, gt))  # ssim是高越好，那么对应的loss是越低越好
        return loss
