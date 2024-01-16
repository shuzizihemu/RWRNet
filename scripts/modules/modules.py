# coding=utf-8
"""
    @Project: RWRNet_Paddle
    @Author：JFZ
    @File： modules.py
    @Date：2023/11/21 17:16
    @EnvConfig: pytorch 1.12.1 + cu116
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
from scripts.modules.swinT import SwinT


def conv3x3(inchannel, outchannel, bias=True):
    conv = nn.Conv2D(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias_attr=bias)
    return conv


def conv_down(inchannel, outchannel, bias=False):
    conv = nn.Conv2D(inchannel, outchannel, kernel_size=4, stride=2, padding=1, bias_attr=bias)
    return conv


def conv(inchannel, outchannel, kernel_size, stride=1, bias=False, ):
    conv = nn.Conv2D(inchannel, outchannel, kernel_size, stride=stride, padding=(kernel_size // 2), bias_attr=bias)
    return conv


class SAM(nn.Layer):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = F.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class CBAM_Module(nn.Layer):
    def __init__(self, channels, reduction=16, bias_attr=True):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2D(output_size=1)
        self.fc1 = nn.Conv2D(in_channels=channels, out_channels=channels // reduction, kernel_size=1, padding=0,
                             bias_attr=bias_attr)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(in_channels=channels // reduction, out_channels=channels, kernel_size=1, padding=0,
                             bias_attr=bias_attr)

        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3,
                                           bias_attr=bias_attr)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention Module
        module_input = x
        avg = self.relu(self.fc1(self.avg_pool(x)))
        avg = self.fc2(avg)
        mx = self.relu(self.fc1(self.max_pool(x)))
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)

        # Spatial Attention Module
        x = module_input * x
        module_input = x
        avg = paddle.mean(x, axis=1, keepdim=True)
        mx = paddle.argmax(x, axis=1, keepdim=True)
        mx = paddle.cast(mx, 'float32')
        x = paddle.concat([avg, mx], axis=1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x

        return x


class UNetUpBlock(nn.Layer):
    def __init__(self, inchannel, outchannel, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Conv2DTranspose(inchannel, outchannel, kernel_size=2, stride=2,
                                     bias_attr=True)  # 上采样卷积，尺寸扩大一倍，通道减少一半
        self.conv_block = UNetConvBlock(inchannel, outchannel, downsample=False,
                                        relu_slope=relu_slope)  # 双卷积，这里的输入通道 = concat(up特征，桥接特征)

    def forward(self, x, bridge):
        up = self.up(x)
        out = paddle.concat((up, bridge), axis=1)
        out = self.conv_block(out)
        return out


class RFFModule(nn.Layer):
    def __init__(self, outchannel):
        super(RFFModule, self).__init__()
        self.rff_enc = CBAM_Module(outchannel, bias_attr=True)
        self.rff_dec = CBAM_Module(outchannel, bias_attr=True)

    def forward(self, x, enc, dec):
        return x + self.rff_enc(enc) + self.rff_dec(dec)


class UNetConvBlock(nn.Layer):
    def __init__(self, inchannel, outchannel, downsample, relu_slope, use_rff=False, use_IN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2D(inchannel, outchannel, kernel_size=1, stride=1, padding=0,
                                  bias_attr=True)
        self.use_rff = use_rff
        self.use_IN = use_IN

        self.conv1 = nn.Conv2D(inchannel, outchannel, kernel_size=3, padding=1, bias_attr=True)
        self.relu1 = nn.LeakyReLU(relu_slope)
        self.conv2 = nn.Conv2D(outchannel, outchannel, kernel_size=3, padding=1, bias_attr=True)
        self.relu2 = nn.LeakyReLU(relu_slope)

        if downsample and use_rff:
            self.rffModule = RFFModule(outchannel)
        if use_IN:
            self.norm = nn.InstanceNorm2D(outchannel // 2)
        if downsample:
            self.downsample = conv_down(outchannel, outchannel, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv1(x)
        if self.use_IN:
            feature1, feature2 = paddle.chunk(out, chunks=2, axis=1)
            out = paddle.concat((self.norm(feature1), feature2), axis=1)
        out = self.relu1(out)
        out = self.relu2(self.conv2(out))
        out = out + self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_rff
            out = self.rffModule(out, enc, dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class GLFEBlock(nn.Layer):
    def __init__(self, inchannel, outchannel, downsample, relu_slope, use_rff=False, use_IN=True, patch_size=128):
        super().__init__()
        self.downsample = downsample
        self.identity = nn.Conv2D(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias_attr=True)
        self.use_rff = use_rff
        self.use_IN = use_IN

        self.swin1 = SwinT(inchannel, outchannel, input_resolution=(patch_size, patch_size), num_heads=8,
                           window_size=8, )
        self.relu1 = nn.LeakyReLU(relu_slope)
        self.conv2 = nn.Conv2D(outchannel, outchannel, kernel_size=3, padding=1, bias_attr=True)
        self.relu2 = nn.LeakyReLU(relu_slope)

        self.rffModule = RFFModule(outchannel)

        if use_IN:
            self.norm = nn.InstanceNorm2D(outchannel // 2)

        if downsample:
            self.downsample = conv_down(outchannel, outchannel, bias=False)

    def forward(self, x, enc=None, dec=None):
        out1 = self.swin1(x)
        
        feature1, feature2 = paddle.chunk(out1, chunks=2, axis=1)
        out = paddle.concat((self.norm(feature1), feature2), axis=1)
        out = self.relu1(out)
        out = self.relu2(self.conv2(out))
        out = out + self.identity(out1)
        if enc is not None and dec is not None:
            assert self.use_rff
            out = self.rffModule(out, enc, dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out
