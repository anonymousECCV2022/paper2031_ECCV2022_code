import math
from torch import nn
import torch.nn.init as init
from .common import AdaptiveFM
from .dgr import DGR
import torch
import torch.nn.functional as F

def make_model(args, parent=False):
    return ESPCN(args)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def espcn_conv(input,prefix, weights_dict):
    x = F.conv2d(input, weights_dict[prefix + '.weight'], weights_dict[prefix + '.bias'], stride=1, padding=1)
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
        if self.args.new_espcn:
            self.conv1 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(64, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
        else:
            self.conv1 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(self.scale)
        
        self._initialize_weights()


    def forward(self, x, num, weights_dict=None):
        if weights_dict!=None:
            #print(3333333333333333333333)
            out = self.act_func(espcn_conv(x, 'model.conv1', weights_dict))
            out = self.act_func(espcn_conv(out, 'model.conv2', weights_dict))
            out = self.act_func(espcn_conv(out, 'model.conv3', weights_dict))
            out = self.pixel_shuffle(espcn_conv(out, 'model.conv4', weights_dict))
            return out
        else:
            #print(1111111111111111111111)
            if self.args.new_espcn:
                res = self.act_func(self.conv1(x))
                out = self.act_func(self.conv2(res))
                out = self.conv3(out)
                out = res+out
                out = self.pixel_shuffle(self.conv4(out))
                return out
            else:
                out = self.act_func(self.conv1(x))
                out = self.act_func(self.conv2(out))
                out = self.act_func(self.conv3(out))
                out = self.pixel_shuffle(self.conv4(out))
                return out

    def _initialize_weights(self):
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