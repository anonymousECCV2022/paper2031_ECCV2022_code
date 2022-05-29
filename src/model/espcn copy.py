import math
from torch import nn
import torch.nn.init as init
from .common import AdaptiveFM
from .dgr import DGR
import torch

def make_model(args, parent=False):
    return ESPCN(args)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

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
        self.adafm = args.adafm
        self.use_adafm = args.use_adafm
        self.segnum = args.segnum
        self.adafm_espcn = args.adafm_espcn
        self.edsr_espcn = args.edsr_espcn
        self.sidetuning = args.sidetuning
        self.adafm_side = args.adafm_side
        self.adafm_k = 7

        if self.adafm_espcn:
            self.AdaFM1 = AdaptiveFM(64,self.adafm_k)
            self.AdaFM11 = AdaptiveFM(64,self.adafm_k)
            self.AdaFM2 = AdaptiveFM(64,self.adafm_k)
            self.AdaFM22 = AdaptiveFM(64,self.adafm_k)
            self.AdaFM3 = AdaptiveFM(32,self.adafm_k)
            self.AdaFM33 = AdaptiveFM(32,self.adafm_k)
            self.AdaFM4 = AdaptiveFM(3 * (self.scale ** 2),self.adafm_k)
            self.conv1 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
            self.pixel_shuffle = nn.PixelShuffle(self.scale)

            self._initialize_weights()

        elif self.edsr_espcn or self.sidetuning or self.adafm_side:
            self.st_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            #self.dgr1 = DGR(256,4)
            self.st_conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.st_conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.st_conv4 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1))
            
            #self.st_conv0 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))
            # self.st_conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            # self.st_conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            # self.st_conv3 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
            # self.pixel_shuffle = nn.PixelShuffle(self.scale)

            # self._initialize_weights()
        
        else:
            self.conv1 = nn.Conv2d(self.n_colors, 64, (3, 3), (1, 1), (1, 1))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, 3 * (self.scale ** 2), (3, 3), (1, 1), (1, 1))
            self.pixel_shuffle = nn.PixelShuffle(self.scale)
            
            self._initialize_weights()

            if self.adafm:
                if self.use_adafm:
                    self.adafms1 = nn.ModuleList([AdaptiveFM(64,1) for _ in range(self.segnum)])
                    self.adafms2 = nn.ModuleList([AdaptiveFM(64,1) for _ in range(self.segnum)])
                    self.adafms3 = nn.ModuleList([AdaptiveFM(32,1) for _ in range(self.segnum)])


    def forward(self, x, num):
        if self.adafm_espcn:
            out = self.AdaFM1(self.conv1(x))
            out = self.act_func(out)
            out = self.AdaFM11(out)
            out = self.AdaFM2(self.conv2(out))
            out = self.act_func(out)
            out = self.AdaFM22(out)
            out = self.AdaFM3(self.conv3(out))
            out = self.act_func(out)
            out = self.AdaFM33(out)
            out = self.AdaFM4(self.conv4(out))
            out = self.pixel_shuffle(out)
            return out
        elif self.edsr_espcn or self.sidetuning or self.adafm_side:
            out = self.st_conv1(x)
            #out = self.dgr1(out)
            out = self.st_conv2(out)
            out = self.st_conv3(out)
            out = self.st_conv4(out)
            
            # out = self.act_func(self.st_conv0(x))
            # out = self.act_func(self.st_conv1(x))
            # out = self.act_func(self.st_conv2(out))
            # out = self.pixel_shuffle(self.st_conv3(out))
            return out
        elif self.adafm:
            out = self.act_func(self.conv1(x))
            if self.use_adafm:
                out = self.adafms1[num](out)
            out = self.act_func(self.conv2(out))
            if self.use_adafm:
                out = self.adafms2[num](out)
            out = self.act_func(self.conv3(out))
            if self.use_adafm:
                out = self.adafms3[num](out)
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

        #init.orthogonal_(self.st_conv0.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.st_conv1.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.st_conv2.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.st_conv3.weight)

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)