"""
#-*- coding = utf-8 -*- 
#@Time: 2023 03 09  下午9:36
#@Author:JFZ
#@File：test_metric.py
#@Software: PyCharm
"""
import os
from config import Config
import random
import numpy as np
import paddle
from paddle.io import DataLoader
from paddle import nn
from scripts.utils.utils import mkdirs, mkdir, current_time, TXTLogs
from scripts.models.models import RWRNet
from scripts.utils.metrices import batch_SSIM, batch_PSNR, compute_RMSE
from datasets.SDIWRD import LoadSDIWRD
from datasets.utils import ToNumpy, showImages, showImage, concat_image
import shutil
from PIL import Image


class Test(object):
    def __init__(self, opt, iter=2, dilation=False):
        self.opt = opt
        # 固定随机种子
        self.set_seed(opt.SEED)
        # 设置程序所需文件夹
        self.config_dirs()
        # 加载模型
        self.model = RWRNet(iter=iter, dilation=dilation)
        # 设置优化器
        # 开始的一些设置，以防影响resume
        self.start_epoch = opt.TRAINING.START_EPOCH
        self.end_epoch = opt.TRAINING.END_EPOCH
        self.print_freq = opt.TRAINING.PRINT_FREQ
        # 恢复训练？
        if opt.TRAINING.RESUME:
            if self.opt.TRAINING.RESUME_PATH is None:
                ckpt_path = os.path.join(self.model_dir, "model_best.pdparams")
            else:
                ckpt_path = self.opt.TRAINING.RESUME_PATH
            self.resume(path=ckpt_path)
            # 损失函数
        self.criterion = nn.L1Loss()  # 损失函数
        # 数据加载
        val_dir = opt.TRAINING.VAL_DIR

        self.testsets = LoadSDIWRD(val_dir, train=False)  # 测试集
        self.testsets_len = len(self.testsets)
        self.testsets = DataLoader(self.testsets, batch_size=1, shuffle=False, num_workers=4)

    def val(self):
        self.test(epoch=0)

    @paddle.no_grad()
    def test(self, epoch):
        self.model.eval()
        info = "test epoch {}\t~~~~~~~~~~".format(epoch)
        print(info)

        psnr_list = []
        ssim_list = []
        rmse_list = []
        rmsew_list = []
        for i, (wm, gt, mask) in enumerate(self.testsets):
            out_refine = self.model(wm)[-1]
            out_refine = paddle.clip(out_refine, 0.0, 1.0)
            current_psnr = batch_PSNR(out_refine, gt, data_range=1.0)
            psnr_list.append(current_psnr)
            current_ssim = batch_SSIM(out_refine, gt)
            ssim_list.append(current_ssim)
            current_rmse = compute_RMSE(out_refine, gt, mask, is_w=False)
            rmse_list.append(current_rmse)
            current_rmsew = compute_RMSE(out_refine, gt, mask, is_w=True)
            rmsew_list.append(current_rmsew)

            if (i + 1) % self.print_freq == 0:
                info = "eval {} | current_psnr {:.2f}| current_ssim {:.2f}% | time {}".format(
                    epoch, current_psnr, current_ssim * 100, current_time()
                )
                print(info)

        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        avg_rmse = sum(rmse_list) / len(rmse_list)
        avg_rmsew = sum(rmsew_list) / len(rmsew_list)
        info = "eval {} | avg_psnr {:.2f}| avg_ssim {:.2f}% | avg_rmse {:.2f}| avg_rmsew {:.2f} | time {}".format(
            epoch, avg_psnr, avg_ssim * 100, avg_rmse, avg_rmsew,
            current_time()
        )
        print(info)

        return avg_psnr, avg_ssim, avg_rmse, avg_rmsew

    def resume(self, path):
        ckpt = paddle.load(path)
        self.start_epoch = ckpt["epoch"] + 1  # 要从当前加载的下一轮开始
        self.best_psnr = ckpt["best_psnr"]  # 这个也必须加载，否则低的psnr可能会覆盖掉之前的psnr
        self.model.set_state_dict(ckpt['state_dict'])
        print("从{}加载权重成功！！\n最佳avg_psnr epoch = {} 当前继续训练的epoch = {}".format(path, ckpt["epoch"], self.start_epoch, ))

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def config_dirs(self):
        self.save_root = self.opt.TRAINING.SAVE_ROOT
        self.experment_name = self.opt.MODEL.EXPERMENT_NAME
        self.result_dir = os.path.join(self.save_root, self.experment_name, "results")
        self.model_dir = os.path.join(self.save_root, self.experment_name, "models")
        mkdirs(paths=[self.result_dir, self.model_dir])


if __name__ == '__main__':
    opt = Config("test.yml")
    print(opt)
    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    method = Test(opt=opt)
    method.val()
