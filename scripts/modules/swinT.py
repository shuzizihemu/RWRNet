

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DropPath(nn.Layer):

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor()  # mask
        output = inputs.divide(keep_prob) * random_tensor  # divide is to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        w_attr_1, b_attr_1 = self._init_weights()
        self.reduction = nn.Linear(4 * dim,
                                   2 * dim,
                                   weight_attr=w_attr_1,
                                   bias_attr=False)

        w_attr_2, b_attr_2 = self._init_weights_layernorm()
        self.norm = nn.LayerNorm(4 * dim,
                                 weight_attr=w_attr_2,
                                 bias_attr=b_attr_2)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward(self, x):
        h, w = self.input_resolution
        b, _, c = x.shape
        x = x.reshape([b, h, w, c])
        x = x.transpose([0, 3, 1, 2])
        x = F.interpolate(x, size=[i // 2 * 2 for i in x.shape[-2:]], mode="BILINEAR")
        x = x.transpose([0, 2, 3, 1])
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = paddle.concat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.reshape([b, -1, 4 * c])  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features, dropout):
        super(Mlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def windows_partition(x, window_size):
    """ partite windows into window_size x window_size
    Args:
        x: Tensor, shape=[b, h, w, c]
        window_size: int, window size
    Returns:
        x: Tensor, shape=[num_windows*b, window_size, window_size, c]
    """

    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size,
                   C])  # [bs,num_window,window_size,num_window,window_size,C]
    x = x.transpose([0, 1, 3, 2, 4, 5])  # [bs,num_window,num_window,window_size,window_Size,C]
    x = x.reshape([-1, window_size, window_size, C])  # (bs*num_windows,window_size, window_size, C)

    return x


def windows_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size,
                         -1])  # [bs,num_window,num_window,window_size,window_Size,C]
    x = x.transpose([0, 1, 3, 2, 4, 5])  # [bs,num_window,window_size,num_window,window_size,C]
    x = x.reshape([B, H, W, -1])  # (bs,num_windows*window_size, num_windows*window_size, C)
    return x


class WindowAttention(nn.Layer):
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5

        self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        # relative position index for each token inside window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # [2, window_h, window_w]
        coords_flatten = paddle.flatten(coords, 1)  # [2, window_h * window_w]
        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(dim,
                             dim * 3,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if qkv_bias else False)

        self.attn_dropout = nn.Dropout(attention_dropout)

        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(dim,
                              dim,
                              weight_attr=w_attr_2,
                              bias_attr=b_attr_2)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def get_relative_pos_bias_from_pos_index(self):
        # relative_position_bias_table is a ParamBase object
        # https://github.com/PaddlePaddle/Paddle/blob/067f558c59b34dd6d8626aad73e9943cf7f5960f/python/paddle/fluid/framework.py#L5727
        table = self.relative_position_bias_table  # N x num_heads
        # index is a tensor
        index = self.relative_position_index.reshape([-1])  # window_h*window_w * window_h*window_w
        # NOTE: paddle does NOT support indexing Tensor by a Tensor
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias

    def forward(self, x, mask=None):
        qkv = self.qkv(x).chunk(3, axis=-1)  # {list:3}
        q, k, v = map(self.transpose_multihead, qkv)

        q = q * self.scale

        attn = paddle.matmul(q, k, transpose_y=True)  # SwinV2,修改此处为余弦注意力

        relative_position_bias = self.get_relative_pos_bias_from_pos_index()

        relative_position_bias = relative_position_bias.reshape(
            [self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1],
             -1])

        # nH, window_h*window_w, window_h*window_w
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(
                [x.shape[0] // nW, nW, self.num_heads, x.shape[1], x.shape[1]])
            attn += mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, x.shape[1], x.shape[1]])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        new_shape = z.shape[:-2] + [self.dim]
        z = z.reshape(new_shape)
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


class SwinTransformerBlock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.,
                 attention_dropout=0., droppath=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        w_attr_1, b_attr_1 = self._init_weights_layernorm()
        self.norm1 = nn.LayerNorm(dim,
                                  weight_attr=w_attr_1,
                                  bias_attr=b_attr_1)

        self.attn = WindowAttention(dim,
                                    window_size=(self.window_size, self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attention_dropout=attention_dropout,
                                    dropout=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else None

        w_attr_2, b_attr_2 = self._init_weights_layernorm()
        self.norm2 = nn.LayerNorm(dim,
                                  weight_attr=w_attr_2,
                                  bias_attr=b_attr_2)

        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = paddle.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = windows_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape((-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = paddle.where(attn_mask != 0,
                                     paddle.ones_like(attn_mask) * float(-100.0),
                                     attn_mask)
            attn_mask = paddle.where(attn_mask == 0,
                                     paddle.zeros_like(attn_mask),
                                     attn_mask)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def _init_weights_layernorm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        h = x

        new_shape = [B, H, W, C]
        x = x.reshape(new_shape)  # [bs,H,W,C]

        if self.shift_size > 0:
            shifted_x = paddle.roll(x,
                                    shifts=(-self.shift_size, -self.shift_size),
                                    axis=(1, 2))  # [bs,H,W,C]
        else:
            shifted_x = x

        x_windows = windows_partition(shifted_x, self.window_size)  # [bs*num_windows,7,7,C]
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])  # [bs*num_windows,7*7,C]

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [bs*num_windows,7*7,C]
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])  # [bs*num_windows,7,7,C]

        shifted_x = windows_reverse(attn_windows, self.window_size, H, W)  # [bs,H,W,C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(shifted_x,
                            shifts=(self.shift_size, self.shift_size),
                            axis=(1, 2))
        else:
            x = shifted_x

        x = x.reshape([B, H * W, C])  # [bs,H*W,C]
        x = self.norm1(x)  # [bs,H*W,C]    #移到这里

        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x
        h = x  # [bs,H*W,C]

        '''
        SwinV2,将此处修改为后归一化
        x = self.norm2(x)       # [bs,H*W,C]
        '''

        x = self.mlp(x)  # [bs,H*W,C]
        x = self.norm2(x)  # 放在这里

        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x
        return x


class SwinT(nn.Layer):
    

    def __init__(self, in_channels, out_channels, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.,
                 attention_dropout=0., droppath=0., downsample=False):
        super().__init__()
        self.dim = in_channels
        self.window_size = window_size
        self.input_resolution = tuple([i // self.window_size * self.window_size for i in input_resolution])

        self.blocks = nn.LayerList()
        for i in range(2):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=in_channels, input_resolution=self.input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    dropout=dropout, attention_dropout=attention_dropout,
                    droppath=droppath[i] if isinstance(droppath, list) else droppath))

        self.cnn = nn.Conv2D(in_channels=in_channels * 2 if downsample else in_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             stride=1,
                             )

        if downsample:
            self.downsample = PatchMerging(self.input_resolution, dim=in_channels)
        else:
            self.downsample = None

    def forward(self, x):
        x = F.interpolate(x=x, size=self.input_resolution, mode="BILINEAR")

        B, C, H, W = x.shape
        x = x.reshape([B, C, H * W])
        x = x.transpose((0, 2, 1))  # [B, H*W, C]

        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
            x = x.transpose((0, 2, 1))  # [B, C * 2, H//2 * W//2]
            x = x.reshape([B, C * 2, H // 2, W // 2])
        else:
            x = x.transpose((0, 2, 1))
            x = x.reshape([B, C, H, W])
        x = self.cnn(x)
        return x
