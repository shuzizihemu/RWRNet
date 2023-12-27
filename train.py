# coding=utf-8
"""
    @Project: RWRNet_Paddle
    @Author：JFZ
    @File： train.py
    @Date：2023/11/21 16:58
    @EnvConfig: pytorch 1.12.1 + cu116
"""
import shutil
import random
import numpy as np
from PIL import Image

import paddle
from paddle import nn
import paddle.optimizer as optim
from paddle.io import DataLoader

from datasets.SDIWRD import LoadSDIWRD
from scripts.utils import *
from scripts.models.models import RWRNet
from scripts.LossFunction import SSIMLoss
from scripts.metric import batch_PSNR, batch_SSIM

from config import Config

class Trainer(object):
    def __init__(self, opt, iter=2):
        self.opt = opt
        # 固定种子
        self.set_seed(self.opt.SEED)

        # 初始化模型
        self.model = RWRNet(iter=iter)

        # 初始化优化器
        self.set_scheduler()

        # 设置文件夹
        self.config_dirs()
        self.save_images = opt.TRAINING.SAVE_IMAGES  # 保存验证时的结果
        if self.save_images:
            self.pred_image_dir = os.path.join(self.result_dir, "images")  # 保存每轮结果的文件夹！！！
            self.save_best_pred_image_dir = os.path.join(self.result_dir, "best_images")  # 到时候将预测的最好的结果，复制到这个文件夹中
            mkdirs([self.pred_image_dir, self.save_best_pred_image_dir])

        # 初始化记录变量
        self.start_epoch = opt.TRAINING.START_EPOCH
        self.end_epoch = opt.TRAINING.END_EPOCH
        self.print_freq = opt.TRAINING.PRINT_FREQ
        self.global_print_step = 0  # 记录打印次数
        self.global_iter = 0  # 记录迭代次数
        self.best_psnr = 0
        self.best_iter = 0
        self.best_epoch = 0
        self.current_ssim = 0

        # 加载预训练权重继续训练
        if opt.TRAINING.RESUME:
            if self.opt.TRAINING.RESUME_PATH is None:
                ckpt_path = os.path.join(self.model_dir, "model_best.pdparams")
            else:
                ckpt_path = self.opt.TRAINING.RESUME_PATH
            self.resume(path=ckpt_path)

        # 初始化损失函数
        self.criterion = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

        # 初始化数据
        train_dir = opt.TRAINING.TRAIN_DIR
        val_dir = opt.TRAINING.VAL_DIR
        self.trainsets = LoadSDIWRD(train_dir, train=True)  # 训练集
        self.trainsets_len = len(self.trainsets)
        self.trainsets = DataLoader(self.trainsets, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, drop_last=False,
                                    num_workers=0)

        self.testsets = LoadSDIWRD(val_dir, train=False)  # 测试集
        self.testsets_len = len(self.testsets)
        self.testsets = DataLoader(self.testsets, batch_size=1, shuffle=False, num_workers=0)

        # 日志记录
        self.tensorboard = Board(log_save_path=self.log_dir)
        self.txtlogwriter = TXTLogs(file_path=self.log_dir, filename="{}.txt".format(self.experment_name))

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def config_dirs(self):
        self.save_root = self.opt.TRAINING.SAVE_ROOT
        self.experment_name = self.opt.MODEL.EXPERMENT_NAME
        self.result_dir = os.path.join(self.save_root, self.experment_name, "results")
        self.log_dir = os.path.join(self.save_root, self.experment_name, "logs")
        self.model_dir = os.path.join(self.save_root, self.experment_name, "models")
        mkdirs(paths=[self.result_dir, self.log_dir, self.model_dir])

    def set_scheduler(self):
        new_lr = self.opt.OPTIM.LR_INITIAL
        self.scheduler = optim.lr.CosineAnnealingDecay(learning_rate=new_lr, T_max=self.opt.OPTIM.T_MAX, eta_min=1e-6)
        clip_grad_norm = nn.ClipGradByNorm(1e-2)  # 防止模型梯度爆炸
        self.optimizer = optim.Adam(parameters=self.model.parameters(), learning_rate=self.scheduler, weight_decay=.0,
                                    grad_clip=clip_grad_norm)

    def resume(self, path):
        ckpt = paddle.load(path)
        self.start_epoch = ckpt["epoch"] + 1  # 要从当前加载的下一轮开始
        self.best_psnr = ckpt["best_psnr"]  # 这个也必须加载，否则低的psnr可能会覆盖掉之前的psnr
        self.global_iter = ckpt["global_iter"]
        self.global_print_step = ckpt["global_print_step"]
        self.model.set_state_dict(ckpt['state_dict'])
        self.optimizer.set_state_dict(ckpt['optimizer'])
        print("\n\nLoaded weights from {} successfully!\nCurrent training progress: {} epochs trained\nBest average PSNR: {:.2f} dB \nTraining will resume from epoch {}.\n\n".format(path, ckpt["epoch"], self.best_psnr, self.start_epoch))


    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch + 1):
            self.train_one_epoch(epoch)
            avg_psnr, avg_ssim = self.test(epoch)
            self.save_checkpoint(epoch, avg_psnr, avg_ssim)
            self.update_lr()

    def train_one_epoch(self, epoch):
        self.model.train()
        batch = 0
        for i, (wm, gt) in enumerate(self.trainsets):
            out1, out2 = self.model(wm)
            self.total_loss = 0.0
            out1_loss = self.criterion(out1, gt)
            out2_loss = self.criterion(out2, gt)
            ssim_loss = self.ssim_loss(out1, gt)
            ssim_loss += self.ssim_loss(out2, gt)
            self.total_loss = 2 * out2_loss + out1_loss + ssim_loss
            self.optimizer.clear_grad()
            self.total_loss.backward()
            self.optimizer.step()

            batch += self.opt.OPTIM.BATCH_SIZE
            self.global_iter += 1
            if (i + 1) % self.print_freq == 0:
                self.global_print_step += 1
                self.tensorboard.record({
                    "train/total_loss": self.total_loss.item(),
                    "train/out1_loss": out1_loss.item(),
                    "train/out2_loss": out2_loss.item(),
                    "train/ssim_loss": ssim_loss.item(),
                    "train/lr": self.optimizer.get_lr(),
                }, step=self.global_print_step)
                info = "epoch {}/{} | batch {}/{} | lr {:.6f} | total_loss {:.4f} | ssim_loss {:.4f} | out2_loss {:.4f} | out1_loss {:.4f} | iter {} | time {}".format(
                    epoch, self.end_epoch, batch, self.trainsets_len, self.optimizer.get_lr(), self.total_loss.item(),
                    ssim_loss.item(), out2_loss.item(), out1_loss.item(), self.global_iter, current_time()
                )
                print(info)
                self.txtlogwriter.write(info)

    @paddle.no_grad()
    def test(self, epoch):
        self.model.eval()
        info = "test epoch {}\t~~~~~~~~~~".format(epoch)
        print(info)
        self.txtlogwriter.write(info)

        psnr_list = []
        ssim_list = []
        for i, (wm, gt) in enumerate(self.testsets):
            out_refine = self.model(wm)[-1]
            out_refine = paddle.clip(out_refine, 0.0, 1.0)
            current_psnr = batch_PSNR(out_refine, gt, data_range=1.0)
            psnr_list.append(current_psnr)
            current_ssim = batch_SSIM(out_refine, gt)
            ssim_list.append(current_ssim)

            if self.save_images:
                self.save_picture(out_refine, image_id=i + 1)

            if (i + 1) % self.print_freq == 0:
                info = "eval {} | current_psnr {:.2f}| current_ssim {:.2f}% | time {}".format(
                    epoch, current_psnr, current_ssim * 100, current_time()
                )
                print(info)
                self.txtlogwriter.write(info)

        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        self.tensorboard.record({
            "test/avg_psnr": avg_psnr,
            "test/avg_ssim": avg_ssim * 100,
        }, step=epoch)
        info = "eval {} | avg_psnr {:.2f}| avg_ssim {:.2f}% | time {}".format(
            epoch, avg_psnr, avg_ssim * 100, current_time()
        )
        print(info)
        self.txtlogwriter.write(info)

        return avg_psnr, avg_ssim

    def update_lr(self):
        lr_sche = self.optimizer._learning_rate
        lr_sche.step()

    def save_picture(self, predict_image, image_id=1):
        # 将预测的tensor图像保存到文件夹
        predict_image = ToNumpy(predict_image.squeeze(0))[0] * 255
        predict_image = predict_image.astype("uint8")
        predict_image = Image.fromarray(predict_image)
        predict_image.save(os.path.join(self.pred_image_dir, "{}.png".format(image_id)))

    def save_checkpoint(self, epoch, avg_psnr, avg_ssim):
        paddle.save({
            "epoch": epoch,
            "avg_ssim": avg_ssim,
            "best_psnr": self.best_psnr,
            "current_psnr": avg_psnr,
            "global_print_step": self.global_print_step,
            "global_iter": self.global_iter,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path=os.path.join(self.model_dir, "model_last.pdparams"))
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.current_ssim = avg_ssim
            self.epoch = epoch
            self.best_iter = self.global_iter
            paddle.save({
                "best_epoch": self.best_epoch,
                "avg_ssim": avg_ssim,
                "epoch": epoch,
                "global_print_step": self.global_print_step,
                "best_psnr": self.best_psnr,
                "best_iter": self.best_iter,
                "global_iter": self.global_iter,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, path=os.path.join(self.model_dir, "model_best.pdparams"))
            info = "now is psnr is best, save!!!"
            print(info)
            self.txtlogwriter.write(info)
            for i in range(1, self.testsets_len + 1):
                try:
                    shutil.copyfile(
                        os.path.join(self.pred_image_dir, "{}.png".format(i)),
                        os.path.join(self.save_best_pred_image_dir, "{}.png".format(i))
                    )
                except:
                    break

if __name__ == '__main__':
    opt = Config("train_config.yml")
    print(opt)
    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    method = Trainer(opt=opt, iter=2)
    method.train()
