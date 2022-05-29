import math
import torch.nn as nn
import torch.nn.init as init
from .common import AdaptiveFM
from .dgr import DGR
import torch
import torch.nn.functional as F
from model import common


def make_model(args, parent=False):
    return ESPCN(args)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def espcn_conv(input,prefix, weights_dict):
    x = F.conv2d(input, weights_dict[prefix + '.weight'], weights_dict[prefix + '.bias'], stride=1, padding=1)
    return x

def espcn_resblock(input,prefix,weights_dict):
    x = F.conv2d(input, weights_dict[prefix + '.conv1.weight'], weights_dict[prefix + '.conv1.bias'], stride=1, padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights_dict[prefix + '.conv2.weight'], weights_dict[prefix + '.conv2.bias'],stride=1, padding=1)
    x = x + input
    return x

class AdaptiveFM(nn.Module):
    # hello ckx
    def __init__(self, in_channel, kernel_size):

        super(AdaptiveFM, self).__init__()
        padding = get_valid_padding(kernel_size, 1)
        self.transformer = nn.Conv2d(in_channel, in_channel, kernel_size,
                                     padding=padding, groups=in_channel//2)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
    def forward(self, x):
        return self.transformer(x) * self.gamma + x

class ESPCN(nn.Module):
    def __init__(self, args):
        super(ESPCN, self).__init__()
        # self.act_func = nn.LeakyReLU(negative_slope=0.2)
        self.act_func = nn.ReLU(inplace=True)
        self.scale = int(args.scale[0])  # use scale[0]
        self.n_colors = args.n_colors
        self.args = args
        n_feats = args.n_feats
        kernel_size = 3 
        conv=common.default_conv
        # resconv1=[nn.Conv2d(64, 64, kernel_size=3, padding=(kernel_size//2), bias=True)]
        # resconv2=[nn.Conv2d(64, 64, kernel_size=3, padding=(kernel_size//2), bias=True)]
        act = nn.ReLU(True)
        if self.args.espcn_resblock or self.args.espcn_resblock_skip:
            self.conv1 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))
            self.res_block1 = common.ResBlock(conv, n_feats, kernel_size, args, bn=True, act=act, res_scale=args.res_scale)
            self.res_block2 = common.ResBlock(conv, n_feats, kernel_size, args, bn=True, act=act, res_scale=args.res_scale)
            self.conv4 = nn.Conv2d(64, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
            # init.orthogonal_(resconv1.weight, init.calculate_gain('relu'))
            # init.orthogonal_(resconv2.weight, init.calculate_gain('relu'))
            # init.orthogonal_(conv4.weight)
            # self.conv1 = nn.Sequential(*conv1)
            # self.res_block1 = nn.Sequential(*res_block1)
            # self.res_block2 = nn.Sequential(*res_block2)
            # self.conv4 = nn.Sequential(*conv4)
            self.pixel_shuffle = nn.PixelShuffle(self.scale)
        else:
            self.conv1 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
            #print('self.scale',self.scale)
            self.pixel_shuffle = nn.PixelShuffle(self.scale)
            
            self._initialize_weights()
        
        


    def forward(self, x, num, weights_dict=None):
        if weights_dict!=None:
            if self.args.espcn_resblock or self.args.espcn_resblock_skip:
                out = self.act_func(espcn_conv(x, 'model.conv1', weights_dict))
                out = espcn_resblock(out,'model.res_block1',weights_dict)
                out = espcn_resblock(out,'model.res_block2',weights_dict)
                out = self.pixel_shuffle(espcn_conv(out, 'model.conv4', weights_dict))
            #print(3333333333333333333333)
                #print('inner',out.shape)
            else:
                out = self.act_func(espcn_conv(x, 'model.conv1', weights_dict))
                out = self.act_func(espcn_conv(out, 'model.conv2', weights_dict))
                out = self.act_func(espcn_conv(out, 'model.conv3', weights_dict))
                out = self.pixel_shuffle(espcn_conv(out, 'model.conv4', weights_dict))
            return out
        else:
            #print(1111111111111111111111)
            # if self.args.new_espcn:
            #     res = self.act_func(self.conv1(x))
            #     out = self.act_func(self.conv2(res))
            #     out = self.conv3(out)
            #     out = res+out
            #     out = self.pixel_shuffle(self.conv4(out))
            #     return out
            if self.args.espcn_resblock:
                out = self.act_func(self.conv1(x))
                out = self.res_block1(out, num)
                out = self.res_block2(out, num)
                out = self.pixel_shuffle(self.conv4(out))
                #print('outter',out.shape)
                return out
            elif self.args.espcn_resblock_skip:
                out = self.act_func(self.conv1(x))
                res = out
                out = self.res_block1(out, num)
                out = self.res_block2(out, num)
                out += res
                out = self.pixel_shuffle(self.conv4(out))
                return out
            else:
                #print('00000',x.shape)
                out = self.act_func(self.conv1(x))
                #print('11111',out.shape)
                out = self.act_func(self.conv2(out))
                #print('22222',out.shape)
                out = self.act_func(self.conv3(out))
                #print('33333',out.shape)
                out = self.pixel_shuffle(self.conv4(out))
                #print('outterespcn',out.shape)
                return out

    def _initialize_weights(self):
        # if self.args.espcn_resblock or self.args.espcn_resblock_skip:
        #     init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        #     init.orthogonal_(self.resconv1.weight, init.calculate_gain('relu'))
        #     init.orthogonal_(self.resconv2.weight, init.calculate_gain('relu'))
        #     init.orthogonal_(self.conv4.weight)
        # else:
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)