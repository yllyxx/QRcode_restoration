import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DSAM import DSAMBlock
# from models.DSAM_1 import DSAMBlock
# from models.DSAM_2 import DSAMBlock
# from models.DSAM_3 import DSAMBlock
from models.MAN import GroupGLKA
from models.wtconv.wtconv2d import WTconv2d
from models.MAN import duochiduconv
'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


'''
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# (PixelUnShuffle)
# (ConditionalBatchNorm2d)
# concat (ConcatBlock)
# sum (ShortcutBlock)
# resblock (ResBlock)
# Channel Attention (CA) Layer (CALayer)
# Residual Channel Attention Block (RCABlock)
# Residual Channel Attention Group (RCAGroup)
# Residual Dense Block (ResidualDenseBlock_5C)
# Residual in Residual Dense Block (RRDB)
# --------------------------------------------
'''


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


# --------------------------------------------
# conditional batch norm
# https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
# --------------------------------------------
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


# --------------------------------------------
# Concat the output of a submodule to its input
# --------------------------------------------
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + 'concat'


# --------------------------------------------
# sum the output of a submodule to its input
# --------------------------------------------
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res

#简易门注意力机制
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta

class BottleNect(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 63
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)

        ### fca ###
        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        ### sca ###
        x_sca = self.fgm(x_sca)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)
# #简易门控线性单元2
# class SimpleGate(nn.Module):
#     def __init__(self, n_feats):
#         super().__init__()
#         i_feats = n_feats * 2
#
#         self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
#         # self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups= n_feats)
#         self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
#
#         # self.norm = LayerNorm(n_feats, data_format='channels_first')
#         self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
#
#     def forward(self, x):
#         # Ghost Expand
#         x = self.Conv1(x)
#         a, x = torch.chunk(x, 2, dim=1)
#         x = x * a  # self.DWConv1(a)
#         x = self.Conv2(x)
#
#         return x * self.scale
    #简易门注意力机制3
# class SimpleGate(nn.Module):
#         def forward(self, x):
#             x = x * x
#             return x

# #简易门注意力机制2(先拆分通道再逐点相乘)
# class SimpleGate(nn.Module):
#     def __init__(self,c1):
#         super().__init__()
#         c2 = 2*c1
#         self.conv1 = nn.Conv2d(c1,c2,1,1,0)
#         self.conv2 = nn.Conv2d(c1,c1,1,1,0)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x1, x2 = x.chunk(2, dim=1)
#         x = self.conv2(x1*x2)
#         return x
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
#层归一化
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class DualGate(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(DualGate, self).__init__()



        self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.dwconv_5 = nn.Conv2d(dim//2, dim // 2, kernel_size=5, stride=1, padding=2,
                               groups=dim // 2, bias=bias)
        self.dwconv_dilated2_1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, stride=1, padding=2,
                                        groups=dim // 2, bias=bias, dilation=2)


        self.project_out = nn.Conv2d(dim//2, dim//2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)


        x1, x2 = x.chunk(2, dim=1)

        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)

        x = F.mish(x2) * x1

        x = self.project_out(x)


        return x
# # # --------------------------------------------
# # # NAFBLOCK(将1*1，3*3中的3*3变为小波卷积)
# # # --------------------------------------------

# class NAFBlock(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, bias=False,drop_out_rate=0.):
#         super().__init__()
#         dw_channel = c * DW_Expand
#         self.conv2 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=True)
#         # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
#         #                        groups=dw_channel,
#         #                        bias=True)
#         self.conv1 = WTconv2d(c, c, kernel_size=3, wt_levels=3)
#         self.conv3_3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=True)
#         self.conv3 = WTconv2d(c, c, kernel_size=3, wt_levels=3)
#
#         # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=True),
#         )
#
#         # SimpleGate
#         self.sg = SimpleGate()
#
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=True)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=True)
#
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#
#     def forward(self, inp):
#         x = inp
#
#         x = self.norm1(x)
#
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.sg(x)
#         x = x * self.sca(x)
#         x = self.conv3(x)
#         x = self.conv3_3(x)
#
#         x = self.dropout1(x)
#
#         y = inp + x * self.beta
#
#         x = self.conv4(self.norm2(y))
#         x = self.sg(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#
#         return y + x * self.gamma

# #多尺度卷积
# class NAFBlock(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, bias=False,drop_out_rate=0.):
#         super().__init__()
#         # dw_channel = c * DW_Expand
#         # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
#         #                        bias=bias)
#         # self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
#         #                        bias=bias)
#         self.conv1 = duochiduconv(c)
#         self.conv1_1 = nn.Conv2d(c,c,kernel_size=1,padding=0,stride=1,groups=1,bias=bias)
#         # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
#         #                        groups=dw_channel,
#         #                        bias=bias)
#
#         # self.conv3 = nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=3, padding=1, stride=1,
#         #                        groups=1, bias=bias)
#         self.conv3 = duochiduconv(c)
#         self.conv3_1 = nn.Conv2d(c,c,kernel_size=1,padding=0,stride=1,groups=1,bias=bias)
#
#         # # Simplified Channel Attention
#         # self.sca = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d(1),
#         #     nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1,
#         #               groups=1, bias=bias),
#         # )
#         # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=bias),
#         )
#         # self.att = DSAMBlock(in_channel=c)
#         # self.att = GroupGLKA(c)
#         # SimpleGate
#         self.sg = SimpleGate(c)
#
#
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=bias)
#
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#
#     def forward(self, inp):
#         x = inp
#
#         x = self.norm1(x)
#
#         x = self.conv1(x)
#         x = self.conv1_1(x)
#         # x = self.conv2(x)
#         # x = self.sg(x)
#         x = self.sca(x)
#         x = self.sg(x)
#         x = self.conv3(x)
#         x = self.conv3_1(x)
#         x = self.dropout1(x)
#
#         y = inp + x * self.beta
#
#         x = self.conv4(self.norm2(y))
#         x = self.sg(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#
#         return y + x * self.gamma
"""
drunet+simplebaseline(NAF)
"""
# class NAFBlock(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, bias=False,drop_out_rate=0.):
#         super().__init__()
#         # dw_channel = c * DW_Expand
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
#                                bias=bias)
#         # self.conv1_1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
#         #                        bias=bias)
#         #
#         # self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1,
#         #                        groups=c,
#         #                        bias=bias)
#
#         self.conv3 = nn.Conv2d(in_channels=c // 2, out_channels=c, kernel_size=3, padding=1, stride=1,
#                                groups=1, bias=bias)
#         # self.conv3 = duochiduconv(c)
#         # self.conv3_1 = nn.Conv2d(c,c,kernel_size=1,padding=0,stride=1,groups=1,bias=bias)
#
#         # # Simplified Channel Attention
#         # self.sca = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d(1),
#         #     nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1,
#         #               groups=1, bias=bias),
#         # )
#         # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=c//2, out_channels=c//2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=bias),
#         )
#         # self.att = DSAMBlock(in_channel=c)
#         # self.att = GroupGLKA(c)
#         # SimpleGate
#         # self.sg = SimpleGate()
#         #多尺度门控
#         self.sg1 = DualGate(c)
#         self.sg2 = DualGate(2*c)
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=2*c, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=bias)
#
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#
#     def forward(self, inp):
#         x = inp
#
#         x = self.norm1(x)
#
#         x = self.conv1(x)
#         # x = self.conv1_1(x)
#         # x = self.conv2(x)
#         # x = self.sg(x)
#
#         x = self.sg1(x)
#         x = self.sca(x)
#         # x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.dropout1(x)
#
#         y = inp + x * self.beta
#
#         x = self.conv4(self.norm2(y))
#         x = self.sg2(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#
#         return y + x * self.gamma
# # # # --------------------------------------------
# # # # NAFBLOCK(大尺度多核注意力))
# # # # --------------------------------------------
# class NAFBlock(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, bias=False,drop_out_rate=0.):
#         super().__init__()
#         # dw_channel = c * DW_Expand
#         # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
#         #                        bias=bias)
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
#                                bias=bias)
#         # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
#         #                        groups=dw_channel,
#         #                        bias=bias)
#
#         self.conv3 = nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=3, padding=1, stride=1,
#                                groups=1, bias=bias)
#
#         # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=bias),
#         )
#         self.att = GroupGLKA(c)
#         # SimpleGate
#         self.sg = SimpleGate()
#
#
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=bias)
#
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#
#     def forward(self, inp):
#         x = inp
#
#         x = self.norm1(x)
#
#         x = self.conv1(x)
#         # x = self.conv2(x)
#         # x = self.sg(x)
#         x = self.att(x)
#
#         x = self.sg(x)
#         x = self.sca(x)
#         x = self.conv3(x)
#
#         x = self.dropout1(x)
#
#         y = inp + x * self.beta
#
#         x = self.conv4(self.norm2(y))
#         x = self.sg(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#
#         return y + x * self.gamma


# # NAFBlock(1乘1换为3*3,加上原论文的双域条带注意力)
# class NAFBlock(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, bias=False,drop_out_rate=0.):
#         super().__init__()
#         dw_channel = c * DW_Expand
#         # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
#         #                        bias=bias)
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
#                                bias=bias)
#         # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
#         #                        groups=dw_channel,
#         #                        bias=bias)
#
#         self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1,
#                                groups=1, bias=bias)
#
#         # # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=bias),
#         )
#         self.att = DSAMBlock(in_channel=c)
#         # # SimpleGate
#         self.sg = SimpleGate()
#
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=bias)
#
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#
#     def forward(self, inp):
#         x = inp
#
#         x = self.norm1(x)
#
#         x = self.conv1(x)
#         # x = self.conv2(x)
#         # x = self.sg(x)
#         x = self.att(x)
#         x = x*self.sca(x)
#         x = self.conv3(x)
#
#         x = self.dropout1(x)
#
#         y = inp + x * self.beta
#
#         x = self.conv4(self.norm2(y))
#         x = self.sg(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#
#         return y + x * self.gamma
"""
NAFBlock(原始DRUNET,shuangyu)
"""

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, bias=False,drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=bias)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=bias)
        # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
        #                        groups=dw_channel,
        #                        bias=bias)

        self.conv3 = nn.Conv2d(in_channels=c // 2, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=bias)

        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=bias),
        # )
        #双域条带注意力机制
        self.att = DSAMBlock(in_channel=c//2)
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=bias)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=bias)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.sg(x)
        # x = x * self.sca(x)
        x = self.att(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
        # return y

# """
# 1*1用小波卷积替代
# """
# class NAFBlock(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, bias=False,drop_out_rate=0.):
#         super().__init__()
#         dw_channel = c * DW_Expand
#         # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
#         #                        bias=bias)
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
#                                bias=bias)
#         # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
#         #                        groups=dw_channel,
#         #                        bias=bias)
#
#         self.conv3 = nn.Conv2d(in_channels=c//2, out_channels=c, kernel_size=3, padding=1, stride=1,
#                                groups=1, bias=bias)
#
#         # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=bias),
#         )
#
#         # SimpleGate
#         self.sg = SimpleGate()
#
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=bias)
#         # self.conv6 = WTconv2d(c, c, kernel_size=3, wt_levels=3)
#         # self.conv7 = WTconv2d(c, c, kernel_size=3, wt_levels=3)
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#
#     def forward(self, inp):
#         x = inp
#
#         x = self.norm1(x)
#
#         x = self.conv1(x)
#         # x = self.conv2(x)
#
#         x = x * self.sca(x)
#         x = self.sg(x)
#         x = self.conv3(x)
#         x = self.dropout1(x)
#         y = inp + x * self.beta
#         x = self.conv4(self.norm2(y))
#         x = self.sg(x)
#         x = self.conv5(x)
#         x = self.dropout2(x)
#
#         return y + x * self.gamma

#NAFBlock(第一个和第五个卷积层变为3*3)
# class NAFBlock(nn.Module):
#     def __init__(self, c, DW_Expand=2, FFN_Expand=2, bias=False,drop_out_rate=0.):
#         super().__init__()
#         dw_channel = c * DW_Expand
#         # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
#         #                        bias=bias)
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
#                                bias=bias)
#         # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
#         #                        groups=dw_channel,
#         #                        bias=bias)
#
#         self.conv3 = nn.Conv2d(in_channels=c // 2, out_channels=c, kernel_size=3, padding=1, stride=1,
#                                groups=1, bias=bias)
#
#         # Simplified Channel Attention
#         self.sca = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1,
#                       groups=1, bias=bias),
#         )
#
#         # SimpleGate
#         self.sg = SimpleGate()
#
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=bias)
#
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#
#     def forward(self, inp):
#         x = inp
#
#         x = self.norm1(x)
#
#         x = self.conv1(x)
#         # x = self.conv2(x)
#         x = self.sg(x)
#         x = x * self.sca(x)
#         x = self.conv3(x)
#
#         x = self.dropout1(x)
#
#         y = inp + x * self.beta
#
#         x = self.conv4(self.norm2(y))
#         x = self.sg(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#
#         return y + x * self.gamma


# --------------------------------------------
# NAFBLOCK(改为3*3并加入坐标注意力)
# --------------------------------------------
# class NAFBlock(nn.Module):
#     def __init__(self, c,c2wh, DW_Expand=2, FFN_Expand=2, bias=False,drop_out_rate=0.):
#         super().__init__()
#         dw_channel = c * DW_Expand
#         # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
#         #                        bias=bias)
#         self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
#                                bias=bias)
#         # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
#         #                        groups=dw_channel,
#         #                        bias=bias)
#
#         self.conv3 = nn.Conv2d(in_channels=c // 2, out_channels=c, kernel_size=3, padding=1, stride=1,
#                                groups=1, bias=bias)
#
#         # Simplified Channel Attention
#         # self.sca = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d(1),
#         #     nn.Conv2d(in_channels=c // 2, out_channels=c // 2, kernel_size=1, padding=0, stride=1,
#         #               groups=1, bias=bias),
#         # )
#         # self.zuobiaoatt = CoordAtt(inp=c//2, oup=c//2)
#         self.att = MultiSpectralAttentionLayer(c // 2, c2wh, c2wh, reduction=16, freq_sel_method='top16')
#         # SimpleGate
#         self.sg = SimpleGate()
#
#         ffn_channel = FFN_Expand * c
#         self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
#                                groups=1, bias=bias)
#
#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#
#         self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#         self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#         self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
#
#     def forward(self, inp):
#         x = inp
#
#         x = self.norm1(x)
#
#         x = self.conv1(x)
#         # x = self.conv2(x)
#         x = self.sg(x)
#         # x = x * self.sca(x)
#         # x = self.zuobiaoatt(x)
#         x = self.att(x)
#         x = self.conv3(x)
#
#         x = self.dropout1(x)
#
#         y = inp + x * self.beta
#
#         x = self.conv4(self.norm2(y))
#         x = self.sg(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#
#         return y + x * self.gamma

# --------------------------------------------
# 频域通道注意力
# --------------------------------------------
# def get_freq_indices(method):
#     assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
#                       'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
#                       'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
#     num_freq = int(method[3:])
#     if 'top' in method:
#         all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
#                              6, 1]
#         all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
#                              5, 3]
#         mapper_x = all_top_indices_x[:num_freq]
#         mapper_y = all_top_indices_y[:num_freq]
#     elif 'low' in method:
#         all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
#                              3, 4]
#         all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
#                              4, 3]
#         mapper_x = all_low_indices_x[:num_freq]
#         mapper_y = all_low_indices_y[:num_freq]
#     elif 'bot' in method:
#         all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
#                              3, 6]
#         all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
#                              3, 3]
#         mapper_x = all_bot_indices_x[:num_freq]
#         mapper_y = all_bot_indices_y[:num_freq]
#     else:
#         raise NotImplementedError
#     return mapper_x, mapper_y
#
#
# class MultiSpectralAttentionLayer(torch.nn.Module):
#     def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
#         super(MultiSpectralAttentionLayer, self).__init__()
#         self.reduction = reduction
#         self.dct_h = dct_h
#         self.dct_w = dct_w
#
#         mapper_x, mapper_y = get_freq_indices(freq_sel_method)
#         self.num_split = len(mapper_x)
#         mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
#         mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
#         # make the frequencies in different sizes are identical to a 7x7 frequency space
#         # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
#
#         self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         n, c, h, w = x.shape
#         x_pooled = x
#         if h != self.dct_h or w != self.dct_w:
#             x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
#             # If you have concerns about one-line-change, don't worry.   :)
#             # In the ImageNet models, this line will never be triggered.
#             # This is for compatibility in instance segmentation and object detection.
#         y = self.dct_layer(x_pooled)
#
#         y = self.fc(y).view(n, c, 1, 1)
#         return x * y.expand_as(x)
#
#
# class MultiSpectralDCTLayer(nn.Module):
#     """
#     Generate dct filters
#     """
#
#     def __init__(self, height, width, mapper_x, mapper_y, channel):
#         super(MultiSpectralDCTLayer, self).__init__()
#
#         assert len(mapper_x) == len(mapper_y)
#         assert channel % len(mapper_x) == 0
#
#         self.num_freq = len(mapper_x)
#
#         # fixed DCT init
#         self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
#
#         # fixed random init
#         # self.register_buffer('weight', torch.rand(channel, height, width))
#
#         # learnable DCT init
#         # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
#
#         # learnable random init
#         # self.register_parameter('weight', torch.rand(channel, height, width))
#
#         # num_freq, h, w
#
#     def forward(self, x):
#         assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
#         # n, c, h, w = x.shape
#
#         x = x * self.weight
#
#         result = torch.sum(x, dim=[2, 3])
#         return result
#
#     def build_filter(self, pos, freq, POS):
#         result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
#         if freq == 0:
#             return result
#         else:
#             return result * math.sqrt(2)
#
#     def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
#         dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
#
#         c_part = channel // len(mapper_x)
#
#         for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
#             for t_x in range(tile_size_x):
#                 for t_y in range(tile_size_y):
#                     dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
#                                                                                            tile_size_x) * self.build_filter(
#                         t_y, v_y, tile_size_y)
#
#         return dct_filter
# # --------------------------------------------
# # 坐标注意力
# # --------------------------------------------
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#
#         out = identity * a_w * a_h
#
#         return out
# --------------------------------------------
# simplified information multi-distillation block (IMDB)
# x + conv1(concat(split(relu(conv(x)))x3))
# --------------------------------------------
class IMDBlock(nn.Module):
    """
    @inproceedings{hui2019lightweight,
      title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
      author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
      booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
      pages={2024--2032},
      year={2019}
    }
    @inproceedings{zhang2019aim,
      title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
      author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
      booktitle={IEEE International Conference on Computer Vision Workshops},
      year={2019}
    }
    """
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CL', d_rate=0.25, negative_slope=0.05):
        super(IMDBlock, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1 = conv(in_channels, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0], negative_slope=negative_slope)

    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)
        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)
        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res


# --------------------------------------------
# Enhanced Spatial Attention (ESA)
# --------------------------------------------
class ESA(nn.Module):
    def __init__(self, channel=64, reduction=4, bias=True):
        super(ESA, self).__init__()
        #               -->conv3x3(conv21)-----------------------------------------------------------------------------------------+
        # conv1x1(conv1)-->conv3x3-2(conv2)-->maxpool7-3-->conv3x3(conv3)(relu)-->conv3x3(conv4)(relu)-->conv3x3(conv5)-->bilinear--->conv1x1(conv6)-->sigmoid
        self.r_nc = channel // reduction
        self.conv1 = nn.Conv2d(channel, self.r_nc, kernel_size=1)
        self.conv21 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=1)
        self.conv2 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.r_nc, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.max_pool2d(self.conv2(x1), kernel_size=7, stride=3)  # 1/6
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x2 = F.interpolate(self.conv5(x2), (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        x2 = self.conv6(x2 + self.conv21(x1))
        return x.mul(self.sigmoid(x2))
        # return x.mul_(self.sigmoid(x2))


class CFRB(nn.Module):
    def __init__(self, in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1, bias=True, mode='CL', d_rate=0.5, negative_slope=0.05):
        super(CFRB, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = in_channels  # int(in_channels - self.d_nc)

        assert mode[0] == 'C', 'convolutional layer first'

        self.conv1_d = conv(in_channels, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv1_r = conv(in_channels, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv2_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv2_r = conv(self.r_nc, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv3_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.conv3_r = conv(self.r_nc, self.r_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv4_d = conv(self.r_nc, self.d_nc, kernel_size, stride, padding, bias=bias, mode=mode[0])
        self.conv1x1 = conv(self.d_nc*4, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, mode=mode[0])
        self.act = conv(mode=mode[-1], negative_slope=negative_slope)
        self.esa = ESA(in_channels, reduction=4, bias=True)

    def forward(self, x):
        d1 = self.conv1_d(x)
        x = self.act(self.conv1_r(x)+x)
        d2 = self.conv2_d(x)
        x = self.act(self.conv2_r(x)+x)
        d3 = self.conv3_d(x)
        x = self.act(self.conv3_r(x)+x)
        x = self.conv4_d(x)
        x = self.act(torch.cat([d1, d2, d3, x], dim=1))
        x = self.esa(self.conv1x1(x))
        return x


# --------------------------------------------
# Channel Attention (CA) Layer
# --------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y


# --------------------------------------------
# Residual Channel Attention Block (RCAB)
# --------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, negative_slope=0.2):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


# --------------------------------------------
# Residual Channel Attention Group (RG)
# --------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, nb=12, negative_slope=0.2):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding, bias, mode, reduction, negative_slope)  for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)  # self.rg = ShortcutBlock(nn.Sequential(*RG))

    def forward(self, x):
        res = self.rg(x)
        return res + x


# --------------------------------------------
# Residual Dense Block
# style: 5 convs
# --------------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = conv(nc+gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = conv(nc+2*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = conv(nc+3*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv5 = conv(nc+4*gc, nc, kernel_size, stride, padding, bias, mode[:-1], negative_slope)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


# --------------------------------------------
# Residual in Residual Dense Block
# 3x5c
# --------------------------------------------
class RRDB(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(RRDB, self).__init__()

        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


'''
# --------------------------------------------
# NonLocalBlock2D:
# embedded_gaussian
# +W(softmax(thetaXphi)Xg)
# --------------------------------------------
'''


# --------------------------------------------
# non-local block with embedded_gaussian
# https://github.com/AlexHex7/Non-local_pytorch
# --------------------------------------------
class NonLocalBlock2D(nn.Module):
    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='maxpool', negative_slope=0.2):

        super(NonLocalBlock2D, self).__init__()

        inter_nc = nc // 2
        self.inter_nc = inter_nc
        self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias, mode='C'+act_mode)
        self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

        if downsample:
            if downsample_mode == 'avgpool':
                downsample_block = downsample_avgpool
            elif downsample_mode == 'maxpool':
                downsample_block = downsample_maxpool
            elif downsample_mode == 'strideconv':
                downsample_block = downsample_strideconv
            else:
                raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
            self.phi = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
            self.g = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
        else:
            self.phi = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
            self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
