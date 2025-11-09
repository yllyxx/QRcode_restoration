import torch
import torch.nn as nn
import models.basicblock as B
from models.SMFANet_arch import SMFA
from models.basicblock import BottleNect
import numpy as np
from models.wtconv.wtconv2d import WTconv2d
'''
# ====================
# Residual U-Net
# ====================
citation:
@article{zhang2020plug,
title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
journal={arXiv preprint},
year={2020}
}
# ====================
'''


# '''
# 原始结构
# '''
# class UNetRes(nn.Module):
#     def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
#         super(UNetRes, self).__init__()
#
#         self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')
#         # self.wtconv = WTconv2d()
#         # downsample
#         if downsample_mode == 'avgpool':
#             downsample_block = B.downsample_avgpool
#         elif downsample_mode == 'maxpool':
#             downsample_block = B.downsample_maxpool
#         elif downsample_mode == 'strideconv':
#             downsample_block = B.downsample_strideconv
#         else:
#             raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
#
#         # self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         self.m_down1 = B.sequential(*[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)],downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         # self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_down2 = B.sequential(*[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         self.m_down3 = B.sequential(*[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_body  = B.sequential(*[B.NAFBlock(nc[3], bias=bias) for _ in range(nb)])
#
#         # upsample
#         if upsample_mode == 'upconv':
#             upsample_block = B.upsample_upconv
#         elif upsample_mode == 'pixelshuffle':
#             upsample_block = B.upsample_pixelshuffle
#         elif upsample_mode == 'convtranspose':
#             upsample_block = B.upsample_convtranspose
#         else:
#             raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
#
#         self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)])
#         self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)])
#         self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)])
#
#         self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')
#
#     def forward(self, x0):
# #        h, w = x.size()[-2:]
# #        paddingBottom = int(np.ceil(h/8)*8-h)
# #        paddingRight = int(np.ceil(w/8)*8-w)
# #        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
#         x1 = self.m_head(x0)
#
#         x2 = self.m_down1(x1)
#         x3 = self.m_down2(x2)
#         x4 = self.m_down3(x3)
#         # print('x4:', x4.size())
#         x = self.m_body(x4)
#         # print('x:', x.size())
#         # print('x+x4:', (x+x4).size())
#         x = self.m_up3(x+x4)
#         # print(x.size())
#         x = self.m_up2(x+x3)
#
#         x = self.m_up1(x+x2)
#         # print(x.size())
#         x = self.m_tail(x+x1)
# #        x = x[..., :h, :w]
#
#         return x


### SMFNet 1*1 10.2
# class UNetRes(nn.Module):
#     def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=False):
#         super(UNetRes, self).__init__()
#
#         self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')
#         self.smfa1 = SMFA(dim=512)
#         self.smfa2 = SMFA(dim=256)
#         self.smfa3 = SMFA(dim=128)
#         self.smfa4 = SMFA(dim=64)
#         self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv1_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         # downsample
#         if downsample_mode == 'avgpool':
#             downsample_block = B.downsample_avgpool
#         elif downsample_mode == 'maxpool':
#             downsample_block = B.downsample_maxpool
#         elif downsample_mode == 'strideconv':
#             downsample_block = B.downsample_strideconv
#         else:
#             raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
#
#         # self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         self.m_down1 = B.sequential(*[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)],downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         # self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_down2 = B.sequential(*[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         self.m_down3 = B.sequential(*[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_body  = B.sequential(*[B.NAFBlock(nc[3], bias=bias) for _ in range(nb)])
#
#         # upsample
#         if upsample_mode == 'upconv':
#             upsample_block = B.upsample_upconv
#         elif upsample_mode == 'pixelshuffle':
#             upsample_block = B.upsample_pixelshuffle
#         elif upsample_mode == 'convtranspose':
#             upsample_block = B.upsample_convtranspose
#         else:
#             raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
#
#         self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)])
#         self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)])
#         self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)])
#
#         self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')
#
#     def forward(self, x0):
#         #        h, w = x.size()[-2:]
#         #        paddingBottom = int(np.ceil(h/8)*8-h)
#         #        paddingRight = int(np.ceil(w/8)*8-w)
#         #        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
#         x1 = self.m_head(x0)
#
#         x2 = self.m_down1(x1)
#         x3 = self.m_down2(x2)
#         x4 = self.m_down3(x3)
#         # print('x4:', x4.size())
#         x = self.m_body(x4)
#         # print('x:', x.size())
#         # print('x+x4:', (x+x4).size())
#         # x = self.m_up3(self.conv1_1(self.conv1(x4)+self.smfa1(x)))
#         x = self.m_up3(x4 * self.smfa1(self.conv1(x4)) + self.conv1_1(x))
#         # print(x.size())
#         # x = self.m_up2(self.conv2_1(self.conv2(x3)+self.smfa2(x)))
#         x = self.m_up2(x3 * self.smfa2(self.conv2(x3)) + self.conv2_1(x))
#         # x = self.m_up1(self.conv3_1(self.conv3(x2)+self.smfa3(x)))
#         x = self.m_up1(x2 * self.smfa3(self.conv3(x2)) + self.conv3_1(x))
#         # print(x.size())
#         x = self.m_tail(x + x1)
#         # x = self.m_tail(self.conv4_1(self.conv4(x1)+self.smfa4(x)))
#         # x = self.m_tail(x1 * self.smfa4(self.conv4(x1)+self.conv4_1(x)))
#         #        x = x[..., :h, :w]
#
#         return x
"""
smfa end
"""
class UNetRes(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=False):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')
        self.smfa1 = SMFA(dim=512)
        self.smfa2 = SMFA(dim=256)
        self.smfa3 = SMFA(dim=128)
        self.smfa4 = SMFA(dim=64)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=bias)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=bias)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=bias)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=bias)
        self.conv1_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=bias)
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=bias)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=bias)
        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=bias)
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        # self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        self.m_down1 = B.sequential(*[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)],downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        # self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
        # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
        self.m_down2 = B.sequential(*[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
        self.m_down3 = B.sequential(*[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
        self.m_body = B.sequential(*[B.NAFBlock(nc[3], bias=bias) for _ in range(nb)])
        # self.m_body = B.sequential(BottleNect(dim = nc[3]))
        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')

    def forward(self, x0):
        #        h, w = x.size()[-2:]
        #        paddingBottom = int(np.ceil(h/8)*8-h)
        #        paddingRight = int(np.ceil(w/8)*8-w)
        #        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        x1 = self.m_head(x0)

        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        # print('x4:', x4.size())
        x = self.m_body(x4)
        # print('x:', x.size())
        # print('x+x4:', (x+x4).size())
        # x = self.m_up3(self.conv1_1(self.conv1(x4)+self.smfa1(x)))
        x = self.m_up3(x4 * self.smfa1(self.conv1(x4)) + self.conv1_1(x))
        # print(x.size())
        # x = self.m_up2(self.conv2_1(self.conv2(x3)+self.smfa2(x)))
        x = self.m_up2(x3 * self.smfa2(self.conv2(x3)) + self.conv2_1(x))
        # x = self.m_up1(self.conv3_1(self.conv3(x2)+self.smfa3(x)))
        x = self.m_up1(x2 * self.smfa3(self.conv3(x2)) + self.conv3_1(x))
        # print(x.size())
        x = self.m_tail(x * self.smfa4(self.conv4(x)) + self.conv4_1(x1))

        # x = self.m_tail(self.conv4_1(self.conv4(x1)+self.smfa4(x)))
        # x = self.m_tail(x1 * self.smfa4(self.conv4(x1)+self.conv4_1(x)))
        #        x = x[..., :h, :w]

        return x
# #smfa跳跃连接
# class UNetRes(nn.Module):
#     def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
#         super(UNetRes, self).__init__()
#
#         self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')
#         self.smfa1 = SMFA(dim=512)
#         self.smfa2 = SMFA(dim=256)
#         self.smfa3 = SMFA(dim=128)
#         self.smfa4 = SMFA(dim=64)
#         self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         # downsample
#         if downsample_mode == 'avgpool':
#             downsample_block = B.downsample_avgpool
#         elif downsample_mode == 'maxpool':
#             downsample_block = B.downsample_maxpool
#         elif downsample_mode == 'strideconv':
#             downsample_block = B.downsample_strideconv
#         else:
#             raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
#
#         # self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         self.m_down1 = B.sequential(*[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)],downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         # self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_down2 = B.sequential(*[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         self.m_down3 = B.sequential(*[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_body  = B.sequential(*[B.NAFBlock(nc[3], bias=bias) for _ in range(nb)])
#
#         # upsample
#         if upsample_mode == 'upconv':
#             upsample_block = B.upsample_upconv
#         elif upsample_mode == 'pixelshuffle':
#             upsample_block = B.upsample_pixelshuffle
#         elif upsample_mode == 'convtranspose':
#             upsample_block = B.upsample_convtranspose
#         else:
#             raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
#
#         self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)])
#         self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)])
#         self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)])
#
#         self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')
#
#     def forward(self, x0):
# #        h, w = x.size()[-2:]
# #        paddingBottom = int(np.ceil(h/8)*8-h)
# #        paddingRight = int(np.ceil(w/8)*8-w)
# #        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
#         x1 = self.m_head(x0)
#
#         x2 = self.m_down1(x1)
#         x3 = self.m_down2(x2)
#         x4 = self.m_down3(x3)
#         # print('x4:', x4.size())
#         x = self.m_body(x4)
#         # print('x:', x.size())
#         # print('x+x4:', (x+x4).size())
#         x = self.m_up3(x4+self.smfa1(x))
#         # print(x.size())
#         x = self.m_up2(x3+self.smfa2(x))
#
#         x = self.m_up1(x2+self.smfa3(x))
#         # print(x.size())
#         x = self.m_tail(x1+self.smfa4(x))
# #        x = x[..., :h, :w]
#
#         return x

#smfa进行跳跃连接操作
# #加入layernorm，simplegate与频域注意力机制
# class UNetRes(nn.Module):
#     def __init__(self, c2wh=[16,32,64,128],in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
#         super(UNetRes, self).__init__()
#
#         self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')
#         self.smfa1 = SMFA(dim=512)
#         self.smfa2 = SMFA(dim=256)
#         self.smfa3 = SMFA(dim=128)
#         self.smfa4 = SMFA(dim=64)
#         self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, groups=1,
#                                bias=bias)
#         # downsample
#         if downsample_mode == 'avgpool':
#             downsample_block = B.downsample_avgpool
#         elif downsample_mode == 'maxpool':
#             downsample_block = B.downsample_maxpool
#         elif downsample_mode == 'strideconv':
#             downsample_block = B.downsample_strideconv
#         else:
#             raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
#
#         # self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         self.m_down1 = B.sequential(*[B.NAFBlock(nc[0], c2wh[3],bias=bias) for _ in range(nb)],downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         # self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_down2 = B.sequential(*[B.NAFBlock(nc[1], c2wh[2],bias=bias) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         self.m_down3 = B.sequential(*[B.NAFBlock(nc[2], c2wh[1],bias=bias) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_body  = B.sequential(*[B.NAFBlock(nc[3], c2wh[0],bias=bias) for _ in range(nb)])
#
#         # upsample
#         if upsample_mode == 'upconv':
#             upsample_block = B.upsample_upconv
#         elif upsample_mode == 'pixelshuffle':
#             upsample_block = B.upsample_pixelshuffle
#         elif upsample_mode == 'convtranspose':
#             upsample_block = B.upsample_convtranspose
#         else:
#             raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
#
#         self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.NAFBlock(nc[2], c2wh[1],bias=bias) for _ in range(nb)])
#         self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.NAFBlock(nc[1], c2wh[2],bias=bias) for _ in range(nb)])
#         self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.NAFBlock(nc[0], c2wh[3],bias=bias) for _ in range(nb)])
#
#         self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')
#
#     def forward(self, x0):
# #        h, w = x.size()[-2:]
# #        paddingBottom = int(np.ceil(h/8)*8-h)
# #        paddingRight = int(np.ceil(w/8)*8-w)
# #        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
#         x1 = self.m_head(x0)
#
#         x2 = self.m_down1(x1)
#         x3 = self.m_down2(x2)
#         x4 = self.m_down3(x3)
#         # print('x4:', x4.size())
#         x = self.m_body(x4)
#         # print('x:', x.size())
#         # print('x+x4:', (x+x4).size())
#         x = self.m_up3(self.conv1(x+self.smfa1(x4)))
#         # print(x.size())
#         x = self.m_up2(self.conv2(x+self.smfa2(x3)))
#
#         x = self.m_up1(self.conv3(x+self.smfa3(x2)))
#         # print(x.size())
#         x = self.m_tail(self.conv4(x+self.smfa4(x1)))
# #        x = x[..., :h, :w]
#
#         return x
#加入layernorm，simplegate与简易通道注意力
# class UNetRes(nn.Module):
#     def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
#         super(UNetRes, self).__init__()
#
#         self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')
#
#         # downsample
#         if downsample_mode == 'avgpool':
#             downsample_block = B.downsample_avgpool
#         elif downsample_mode == 'maxpool':
#             downsample_block = B.downsample_maxpool
#         elif downsample_mode == 'strideconv':
#             downsample_block = B.downsample_strideconv
#         else:
#             raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
#
#         # self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         self.m_down1 = B.sequential(*[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)],downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         # self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_down2 = B.sequential(*[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         self.m_down3 = B.sequential(*[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_body  = B.sequential(*[B.NAFBlock(nc[3], bias=bias) for _ in range(nb)])
#
#         # upsample
#         if upsample_mode == 'upconv':
#             upsample_block = B.upsample_upconv
#         elif upsample_mode == 'pixelshuffle':
#             upsample_block = B.upsample_pixelshuffle
#         elif upsample_mode == 'convtranspose':
#             upsample_block = B.upsample_convtranspose
#         else:
#             raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
#
#         self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.NAFBlock(nc[2], bias=bias) for _ in range(nb)])
#         self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.NAFBlock(nc[1], bias=bias) for _ in range(nb)])
#         self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.NAFBlock(nc[0], bias=bias) for _ in range(nb)])
#
#         self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')
#
#     def forward(self, x0):
# #        h, w = x.size()[-2:]
# #        paddingBottom = int(np.ceil(h/8)*8-h)
# #        paddingRight = int(np.ceil(w/8)*8-w)
# #        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
#         x1 = self.m_head(x0)
#
#         x2 = self.m_down1(x1)
#         x3 = self.m_down2(x2)
#         x4 = self.m_down3(x3)
#         # print('x4:', x4.size())
#         x = self.m_body(x4)
#         # print('x:', x.size())
#         # print('x+x4:', (x+x4).size())
#         x = self.m_up3(x+x4)
#         # print(x.size())
#         x = self.m_up2(x+x3)
#
#         x = self.m_up1(x+x2)
#         # print(x.size())
#         x = self.m_tail(x+x1)
# #        x = x[..., :h, :w]
#
#         return x
#加入layernorm，simplegate与频域注意力机制
# class UNetRes(nn.Module):
#     def __init__(self, c2wh=[16,32,64,128],in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=False):
#         super(UNetRes, self).__init__()
#
#         self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')
#
#         # downsample
#         if downsample_mode == 'avgpool':
#             downsample_block = B.downsample_avgpool
#         elif downsample_mode == 'maxpool':
#             downsample_block = B.downsample_maxpool
#         elif downsample_mode == 'strideconv':
#             downsample_block = B.downsample_strideconv
#         else:
#             raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
#
#         # self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         self.m_down1 = B.sequential(*[B.NAFBlock(nc[0], c2wh[3],bias=bias) for _ in range(nb)],downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         # self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         # self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_down2 = B.sequential(*[B.NAFBlock(nc[1], c2wh[2],bias=bias) for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         self.m_down3 = B.sequential(*[B.NAFBlock(nc[2], c2wh[1],bias=bias) for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#         self.m_body  = B.sequential(*[B.NAFBlock(nc[3], c2wh[0],bias=bias) for _ in range(nb)])
#
#         # upsample
#         if upsample_mode == 'upconv':
#             upsample_block = B.upsample_upconv
#         elif upsample_mode == 'pixelshuffle':
#             upsample_block = B.upsample_pixelshuffle
#         elif upsample_mode == 'convtranspose':
#             upsample_block = B.upsample_convtranspose
#         else:
#             raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
#
#         self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.NAFBlock(nc[2], c2wh[1],bias=bias) for _ in range(nb)])
#         self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.NAFBlock(nc[1], c2wh[2],bias=bias) for _ in range(nb)])
#         self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.NAFBlock(nc[0], c2wh[3],bias=bias) for _ in range(nb)])
#
#         self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')
#
#     def forward(self, x0):
# #        h, w = x.size()[-2:]
# #        paddingBottom = int(np.ceil(h/8)*8-h)
# #        paddingRight = int(np.ceil(w/8)*8-w)
# #        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
#         x1 = self.m_head(x0)
#
#         x2 = self.m_down1(x1)
#         x3 = self.m_down2(x2)
#         x4 = self.m_down3(x3)
#         # print('x4:', x4.size())
#         x = self.m_body(x4)
#         # print('x:', x.size())
#         # print('x+x4:', (x+x4).size())
#         x = self.m_up3(x+x4)
#         # print(x.size())
#         x = self.m_up2(x+x3)
#
#         x = self.m_up1(x+x2)
#         # print(x.size())
#         x = self.m_tail(x+x1)
# #        x = x[..., :h, :w]
#
#         return x


"""
原始drunet
"""
# class UNetRes(nn.Module):
#     def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=False):
#         super(UNetRes, self).__init__()
#
#         self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')
#
#         # downsample
#         if downsample_mode == 'avgpool':
#             downsample_block = B.downsample_avgpool
#         elif downsample_mode == 'maxpool':
#             downsample_block = B.downsample_maxpool
#         elif downsample_mode == 'strideconv':
#             downsample_block = B.downsample_strideconv
#         else:
#             raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
#
#         self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
#         self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
#         self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))
#
#         self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
#
#         # upsample
#         if upsample_mode == 'upconv':
#             upsample_block = B.upsample_upconv
#         elif upsample_mode == 'pixelshuffle':
#             upsample_block = B.upsample_pixelshuffle
#         elif upsample_mode == 'convtranspose':
#             upsample_block = B.upsample_convtranspose
#         else:
#             raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
#
#         self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
#         self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
#         self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
#
#         self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')
#
#     def forward(self, x0):
# #        h, w = x.size()[-2:]
# #        paddingBottom = int(np.ceil(h/8)*8-h)
# #        paddingRight = int(np.ceil(w/8)*8-w)
# #        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
#
#         x1 = self.m_head(x0)
#
#         x2 = self.m_down1(x1)
#
#         x3 = self.m_down2(x2)
#
#         x4 = self.m_down3(x3)
#
#         x = self.m_body(x4)
#
#         x = self.m_up3(x+x4)
#
#         x = self.m_up2(x+x3)
#
#         x = self.m_up1(x+x2)
#
#         x = self.m_tail(x+x1)
# #        x = x[..., :h, :w]
#
#
#         return x

if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
    net = UNetRes()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())

# run models/network_unet.py
