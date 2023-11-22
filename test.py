# coding=utf-8
"""
    @Project: RWRNet_Paddle
    @Author：JFZ
    @File： test.py
    @Date：2023/11/21 16:59
    @EnvConfig: pytorch 1.12.1 + cu116
"""
import paddle
from paddle.io import DataLoader
from datasets.SDIWRD_wMask import LoadSDIWRD
from scripts.models.models import RWRNet
from scripts.metric import batch_SSIM, batch_PSNR, compute_RMSE
from scripts.utils import current_time


def run(model_weights, val_dir, print_freq):
    # 1. 获取模型
    model = RWRNet(iter=2)

    # 2. 加载模型权重
    ckpt = paddle.load(model_weights)
    model.set_state_dict(ckpt["state_dict"])

    # 3. 获取数据
    testsets = LoadSDIWRD(val_dir, train=False)  # 测试集
    testsets = DataLoader(testsets, batch_size=1, shuffle=False, num_workers=0)

    # 4. 进行预测
    psnr_list = []
    ssim_list = []
    rmse_list = []
    rmsew_list = []
    for i, (wm, gt, mask) in enumerate(testsets):
        out_refine = model(wm)[-1]
        out_refine = paddle.clip(out_refine, 0.0, 1.0)
        current_psnr = batch_PSNR(out_refine, gt, data_range=1.0)
        psnr_list.append(current_psnr)
        current_ssim = batch_SSIM(out_refine, gt)
        ssim_list.append(current_ssim)
        current_rmse = compute_RMSE(out_refine, gt, mask, is_w=False)
        rmse_list.append(current_rmse)
        current_rmsew = compute_RMSE(out_refine, gt, mask, is_w=True)
        rmsew_list.append(current_rmsew)

        if (i + 1) % print_freq == 0:
            info = "current_psnr {:.2f}| current_ssim {:.2f}% | time {}".format(
                current_psnr, current_ssim * 100, current_time()
            )
            print(info)

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_rmse = sum(rmse_list) / len(rmse_list)
    avg_rmsew = sum(rmsew_list) / len(rmsew_list)
    info = "avg_psnr {:.2f}| avg_ssim {:.2f}% | avg_rmse {:.2f}| avg_rmsew {:.2f} | time {}".format(
        avg_psnr, avg_ssim * 100, avg_rmse, avg_rmsew,
        current_time()
    )
    print(info)


if __name__ == '__main__':
    model_weights = r"output/models/model_best.pdparams"
    val_dir = r"E:\Python_Demo\Datasets\SDIWRD"
    print_freq = 100
    run(model_weights, val_dir, print_freq)
