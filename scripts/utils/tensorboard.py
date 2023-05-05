"""
#-*- coding = utf-8 -*- 
#@Time: 2022 11 08  下午7:41
#@Author:JFZ
#@File：tensorboard.py
#@Software: PyCharm

"""

from visualdl import LogWriter
from scripts.utils.utils import mkdir


class Board():
    def __init__(self, log_save_path="logtest"):
        """Tensorboard
        启动方式 visualdl --logdir checkpoints/log 即可启动
        """
        mkdir(log_save_path)
        self.writer = LogWriter(log_save_path)

    def record(self, dictionaries, step):
        for key, value in dictionaries.items():
            self.writer.add_scalar(tag=key, value=value, step=step)

    def record_image(self, dictionaries, step):
        """图像的格式要为ndarray"""
        for key, value in dictionaries.items():
            self.writer.add_image(tag=key, img=value, step=step)
