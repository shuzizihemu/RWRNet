"""
#-*- coding = utf-8 -*- 
#@Time: 2022 11 12  下午2:49
#@Author:JFZ
#@File：models.py
#@Software: PyCharm
"""

import paddle
from paddle import nn
from scripts.modules.modules import conv3x3, conv_down, conv, SAM, SwintUNetConvBlockV4


class RWRNet(nn.Layer):
    def __init__(self):
        super().__init__()
        # 核心代码将于后续公布
        pass

    def forward(self, x):
        pass
