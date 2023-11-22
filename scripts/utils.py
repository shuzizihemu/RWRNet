# coding=utf-8
"""
    @Project: RWRNet_Paddle
    @Author：JFZ
    @File： utils.py
    @Date：2023/11/21 16:58
    @EnvConfig: pytorch 1.12.1 + cu116
"""
import os
import datetime
from visualdl import LogWriter


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

def mkdirs(paths):
    """批量创建文件夹"""
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """若文件夹不存在，则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


def current_time():
    current_time = datetime.datetime.now()
    current_time = datetime.datetime.strftime(current_time, '%Y-%m-%d %H:%M:%S')  # 2022-11-09 17:06:43
    return current_time


class TXTLogs():
    def __init__(self, file_path="outdir", filename="XXXExpermentlogs.txt"):
        self.filename = os.path.join(file_path, filename)
        with open(self.filename, "a") as f:
            f.write(
                "============================================================={}=============================================================\n".format(
                    current_time()))
        print(self.filename, "创建成功")

    def write(self, content):
        with open(self.filename, "a") as f:
            f.write(content)
            f.write("\n")


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
