import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import miziha


# 定义卷积批归一化块
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):

        # num_channels, 卷积层的输入通道数
        # num_filters, 卷积层的输出通道数
        # stride, 卷积层的步幅
        # groups, 分组卷积的组数，默认groups=1不使用分组卷积

        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = paddle.nn.BatchNorm2D(num_filters)

        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act == 'leaky':
            y = F.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)
        return y


# 定义残差块
# 每个残差块会对输入图片做二次卷积，做一次SwinT，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致

class BottleneckBlock(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 resolution,
                 num_heads=8,
                 window_size=8,
                 downsample=False,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')

        # 创建第二个卷积层 3x3
        # self.conv1 = ConvBNLayer(
        #     num_channels=num_filters,
        #     num_filters=num_filters,
        #     filter_size=3,
        #     stride=stride,
        #     act='relu')

        # 如果尺寸为7x7，启动cnn
        # 使用SwinT进行替换，如下
        if resolution == (7, 7):
            self.swin = ConvBNLayer(num_channels=num_filters,
                                    num_filters=num_filters,
                                    filter_size=3,
                                    stride=1,
                                    act='relu')
        else:
            self.swin = miziha.SwinT(in_channels=num_filters,
                                     out_channels=num_filters,
                                     input_resolution=resolution,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     downsample=downsample)

            # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):

        y = self.conv0(inputs)
        swin = self.swin(y)
        conv2 = self.conv2(swin)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


# 搭建SwinResnet
class SwinResnet(paddle.nn.Layer):
    def __init__(self, num_classes=12):
        super().__init__()

        depth = [3, 4, 6, 3]
        # 残差块中使用到的卷积的输出通道数，图片的尺寸信息，多头注意力参数
        num_filters = [64, 128, 256, 512]
        resolution_list = [[(56, 56), (56, 56)], [(56, 56), (28, 28)], [(28, 28), [14, 14]], [(14, 14), (7, 7)]]
        num_head_list = [4, 8, 16, 32]

        # SwinResnet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        # [3, 224, 224]
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        # [64, 112, 112]
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1)
        # [64, 56, 56]

        # SwinResnet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                # c3、c4、c5将会在第一个残差块使用downsample=True；其余所有残差块downsample=False
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        downsample=True if i == 0 and block != 0 else False,
                        num_heads=num_head_list[block],
                        resolution=resolution_list[block][0] if i == 0 and block != 0 else resolution_list[block][1],
                        window_size=7,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv1 = 1.0 / math.sqrt(2048 * 1.0)
        stdv2 = 1.0 / math.sqrt(256 * 1.0)

        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积和全局池化后，
        # 卷积特征的维度是[B,2048,1,1]，故最后一层全连接的输入维度是2048
        self.out = nn.Sequential(nn.Dropout(0.2),
                                 nn.Linear(in_features=2048, out_features=256,
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.nn.initializer.Uniform(-stdv1, stdv1))),
                                 nn.LayerNorm(256),
                                 nn.Dropout(0.2),
                                 nn.LeakyReLU(),
                                 nn.Linear(in_features=256, out_features=num_classes,
                                           weight_attr=paddle.ParamAttr(
                                               initializer=paddle.nn.initializer.Uniform(-stdv2, stdv2)))
                                 )

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, [y.shape[0], -1])
        y = self.out(y)
        return y
