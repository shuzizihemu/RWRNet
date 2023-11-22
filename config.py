"""
#-*- coding = utf-8 -*- 
#@Time: 2022 11 08  下午6:48
#@Author:JFZ
#@File：config.py
#@Software: PyCharm
"""
from typing import Any, List
from yacs.config import CfgNode as CN


class Config(object):
    def __init__(self, config_yaml: str, config_override: List[Any] = []):
        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False
        self._C.SEED = 2022
        self._C.MODEL = CN()
        self._C.MODEL.EXPERMENT_NAME = 'WatermarkRemoval'
        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 1
        self._C.OPTIM.NUM_EPOCHS = 100
        self._C.OPTIM.T_MAX = 400000
        self._C.OPTIM.NUM_ITERS = 400000
        self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.BETA1 = 0.5
        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.RESUME_PATH = './model_best.pdparams'
        self._C.TRAINING.PRINT_FREQ = 2000
        self._C.TRAINING.SAVE_IMAGES = False
        self._C.TRAINING.TRAIN_DIR = 'images_dir/train'
        self._C.TRAINING.VAL_DIR = 'images_dir/val'
        self._C.TRAINING.SAVE_ROOT = 'checkpoints'
        self._C.TRAINING.TRAIN_PS = 64
        self._C.TRAINING.VAL_PS = 64
        self._C.TRAINING.START_EPOCH = 1
        self._C.TRAINING.END_EPOCH = 100
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        self._C.freeze()

    def dump(self, file_path: str):
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
