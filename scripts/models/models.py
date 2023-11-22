# coding=utf-8
"""
    @Project: RWRNet_Paddle
    @Author：JFZ
    @File： models.py
    @Date：2023/11/21 17:14
    @EnvConfig: pytorch 1.12.1 + cu116
"""
from scripts.modules.modules import *


class RWRNet(nn.Layer):
    def __init__(self, inchannel=3, wf=64, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4, iter=2):
        super().__init__()
        self.depth = depth
        self.iter = iter
        self.conv1 = nn.Conv2D(inchannel, wf, 3, 1, 1, bias_attr=True)

        self.down_path_1 = nn.LayerList()
        patch_size_list = [256, 128, 64, 32, 16]
        prev_channels = wf
        for i in range(depth):  # 0,1,2,3,4
            use_IN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i + 1) < depth else False
            next_channels = (2 ** i) * wf
            if i == 2:
                self.down_path_1.append(
                    GLFEBlock(prev_channels, next_channels, downsample, relu_slope, use_rff=downsample, use_IN=use_IN,
                              patch_size=patch_size_list[i]))
            else:
                self.down_path_1.append(
                    UNetConvBlock(prev_channels, next_channels, downsample, relu_slope, use_rff=downsample,
                                  use_IN=use_IN))
            prev_channels = next_channels

        self.up_path_1 = nn.LayerList()
        self.skip_conv_1 = nn.LayerList()
        for i in reversed(range(depth - 1)):
            next_channels = (2 ** i) * wf
            self.up_path_1.append(UNetUpBlock(prev_channels, next_channels, relu_slope=relu_slope))
            self.skip_conv_1.append(nn.Conv2D(next_channels, next_channels, 3, 1, 1, bias_attr=True))
            prev_channels = next_channels
        self.sam = SAM(prev_channels)
        self.cat = nn.Conv2D(prev_channels * 2, prev_channels, 1, 1, 0, bias_attr=True)
        self.last = conv3x3(prev_channels, inchannel, bias=True)

    def forward(self, x):
        image = x
        sam_feature = None
        current_encs = list()
        current_decs = list()
        pred_list = list()
        for layer in range(self.iter):
            out = self.conv1(image)

            last_encs = current_encs
            current_encs = list()
            last_decs = current_decs
            current_decs = list()

            if layer > 0:
                out = self.cat(paddle.concat((out, sam_feature), axis=1))
                for i, down in enumerate(self.down_path_1):
                    if (i + 1) < self.depth:
                        out, out_up = down(out, last_encs[i], last_decs[-1 - i])
                        current_encs.append(out_up)
                    else:
                        out = down(out)
            else:
                for i, down in enumerate(self.down_path_1):
                    if (i + 1) < self.depth:
                        out, out_up = down(out)
                        current_encs.append(out_up)
                    else:
                        out = down(out)

            for i, up in enumerate(self.up_path_1):
                bridge = self.skip_conv_1[i](current_encs[-1 - i])
                out = up(out, bridge)
                current_decs.append(out)

            if layer < self.iter - 1:
                sam_feature, pred = self.sam(out, image)
            else:
                pred = self.last(out)
                pred = pred + image
            pred_list.append(pred)
        return pred_list
