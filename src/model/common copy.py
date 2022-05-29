import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgr import DGR

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = True

class BasicBlock(nn.Sequential):
    #hello ljm, i am syx
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class BasicBlock_(nn.Module):
    def __init__(
        self, args, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):
        self.use_adafm = args.use_adafm
        super(BasicBlock_, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size, bias=bias)
        if args.adafm:
            if self.use_adafm:
                self.adafms = nn.ModuleList([AdaptiveFM(out_channels,1) for _ in range(args.segnum)])
        self.act = act

    def forward(self, input, num):
        x = self.conv1(input) 
        if self.use_adafm:
            x = self.adafms[num](x)
        x = self.act(x) 
        return x


class AdaptiveFM(nn.Module):
    # hello ckx
    def __init__(self, in_channel, kernel_size):

        super(AdaptiveFM, self).__init__()
        padding = get_valid_padding(kernel_size, 1)
        self.transformer = nn.Conv2d(in_channel, in_channel, kernel_size,
                                     padding=padding, groups=in_channel)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
    def forward(self, x):
        return self.transformer(x) * self.gamma + x
        #print('11111:', self.transformer(x).type())
        #print('22222:', x.type())
        #return self.transformer(x) + x

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, args,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.args = args

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.act = act

    
    def forward(self, input, num):
            x = self.conv1(input)
            x = self.conv2(self.act(x)) 
            res = x.mul(self.res_scale)
            res += input
            return res

class ResBlock_(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, args,
        bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            # if bn:
            #     m.append(AdaptiveFM(n_feats,7))  # the filter size of adafm during finetune. 1 | 3 | 5 | 7
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale 
    
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, kernel_size, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, kernel_size, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, kernel_size, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

