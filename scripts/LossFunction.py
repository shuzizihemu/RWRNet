"""
#-*- coding = utf-8 -*- 
#@Time: 2022 11 18  上午11:11
#@Author:JFZ
#@File：LossFunction.py
#@Software: PyCharm
"""
from paddle import nn
from paddle_ssim import SSIM


class SSIMLoss(nn.Layer):
    def __init__(self):
        super().__init__()
        self.lossfunc = SSIM()

    def forward(self, pred, gt):
        loss = 1 - self.lossfunc(pred, gt)
        return loss
