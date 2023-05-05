"""
#-*- coding = utf-8 -*- 
#@Time: 2022 11 04  下午3:58
#@Author:JFZ
#@File：utils.py
#@Software: PyCharm
"""
import os
import datetime


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
            f.write("============================================================={}=============================================================\n".format(current_time()))
        print(self.filename, "创建成功")

    def write(self, content):
        with open(self.filename, "a") as f:
            f.write(content)
            f.write("\n")
