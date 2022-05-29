import numpy as np
import torch
import torch.nn as nn

from model import common
import torch.nn.functional as F

from option import args
from einops.layers.torch import Rearrange

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}
import pdb


def make_model(args, parent=False):
    return EDSRMLP(args)

def edsr_head(input,prefix, weights_dict):
    x = F.conv2d(input, weights_dict[prefix + '.0.weight'], weights_dict[prefix + '.0.bias'], stride=1, padding=1)
    return x

def edsr_resblock(input,prefix,weights_dict):
    x = F.conv2d(input, weights_dict[prefix + '.conv1.weight'], weights_dict[prefix + '.conv1.bias'], stride=1, padding=1)
    x = F.relu(x)
    x = F.conv2d(x, weights_dict[prefix + '.conv2.weight'], weights_dict[prefix + '.conv2.bias'],stride=1, padding=1)
    x = x + input
    return x

def edsr_conv(input,prefix, weights_dict):
    
    x = F.conv2d(input, weights_dict[prefix + '.weight'], weights_dict[prefix + '.bias'], stride=1, padding=1)
    return x

def edsr_tail(input,prefix,weights_dict):
    x = F.conv2d(input, weights_dict[prefix + '.0.weight'], weights_dict[prefix + '.0.bias'], stride=1, padding=1)
    if args.scale[0]==2:
        ps = nn.PixelShuffle(2)
    else:
        ps = nn.PixelShuffle(3)
    x = ps(x)
    x = F.conv2d(x, weights_dict[prefix + '.2.weight'], weights_dict[prefix + '.2.bias'], stride=1, padding=1)
    return x

def edsr_model(input,num,weight_dict):
    x = edsr_head(input,'model.head',weight_dict)
    for i in range(args.n_resblocks):
        x = edsr_resblock(x,'model.body.{}',weight_dict)
    x = edsr_conv(x, 'model.body.16',weight_dict)
    x = edsr_tail(x,"model.tail.0",weight_dict)
    return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # print('1111',x.shape)
        return self.net(x)

class MlpFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # dim = 64
        # print('dim', dim)
        # print('hidden_dim', hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=(3//2), bias=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=(3//2), bias=True),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
    def forward(self, x):
        # print(x.shape)
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            #nn.LayerNorm(dim),  #dim=64
            #Rearrange('b (h w) d -> b d h w', h=24, w=24),
            #Rearrange('b n d -> b (h w) d', h=24, w=24)
            #FeedForward(num_patch, token_dim, dropout),
            MlpFeedForward(dim, token_dim, dropout),
            
        )

        self.channel_mix = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
            Rearrange('b h w c -> b c h w')
            # Rearrange('b (h w) c -> b c h w',h=24,w=24)
        )
        
    def forward(self, x):
        # print('x1', x.shape)
        # print('x2', self.token_mix(x).shape)
        x = x + self.token_mix(x)
        # print('x2', x.shape)
        x = x + self.channel_mix(x)
    
        #print('x3', x.shape)
        return x

class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        #self.num_patch =  (image_size// patch_size) ** 2  #256
        self.num_patch = 576

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding = 1), #3,64,3,1
            #Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)
        # self.dim_recover = nn.Sequential(
        #     Rearrange('b (h w) c -> b c h w',h=24,w=24)
        # )


    def forward(self, x):

        #print('x0',x.shape)
        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        #x = self.layer_norm(x)
        #print('x1',x.shape)
        #x = x.mean(dim=1)
        # x = self.dim_recover(x)
        #print('x2',x.shape)
        return x

class EDSRMLP(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSRMLP, self).__init__()

        n_resblocks = args.n_resblocks
        self.numbers = n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        self.segnum  = args.segnum
        self.sidetuning = args.sidetuning
        self.adafm_side = args.adafm_side
        self.adafm = args.adafm
        self.maml = args.maml
        self.use_maml = args.use_maml
        self.n_resblocks = args.n_resblocks
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)



        # define head module
        #m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        #m_body = [common.ResBlock(conv, n_feats, kernel_size, bn=True, act=act, res_scale=args.res_scale) for _ in range(n_resblocks)]
        if args.adafm or args.maml:
            m_body = [common.ResBlock(conv, n_feats, kernel_size, args, bn=True, act=act, res_scale=args.res_scale) for _ in range(n_resblocks)]
        elif args.mlp:
            self.body = MLPMixer(in_channels=3, image_size=24, patch_size=1, dim=64, depth=16, token_dim=1024, channel_dim=256)
        else:
            m_body = [common.ResBlock_(conv, n_feats, kernel_size, args, act=act, res_scale=args.res_scale) for _ in range(n_resblocks)]
        #m_body.append(conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        if args.maml:
            if scale == 2:
                m_tail = [
                nn.Conv2d(n_feats, 4*n_feats, kernel_size,padding=(kernel_size//2), bias=True),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, args.n_colors, kernel_size,padding=(kernel_size//2), bias=True)]
                # m_tail = [
                # common.Upsampler(conv, scale, n_feats, kernel_size, act=False),
                # conv(n_feats, args.n_colors, kernel_size)
                # ]
            elif scale == 3:
                m_tail = [
                    nn.Conv2d(n_feats, 9*n_feats, kernel_size,padding=(kernel_size//2), bias=True),
                    nn.PixelShuffle(3),
                    nn.Conv2d(n_feats, args.n_colors, kernel_size,padding=(kernel_size//2), bias=True)]
            elif scale == 4:
                m_tail = [
                    nn.Conv2d(n_feats, 4*n_feats, kernel_size,padding=(kernel_size//2), bias=True),
                    nn.PixelShuffle(2),
                    nn.Conv2d(n_feats, 4*n_feats, kernel_size,padding=(kernel_size//2), bias=True),
                    nn.PixelShuffle(2),
                    nn.Conv2d(n_feats, args.n_colors, kernel_size,padding=(kernel_size//2), bias=True)]
        else:
            m_tail = [
                common.Upsampler(conv, scale, n_feats, kernel_size, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]

        #self.head = nn.Sequential(*m_head)
        # if args.adafm or args.maml:
        #     self.body = nn.ModuleList(m_body)
        # else:
        #     self.body = nn.Sequential(*m_body)
        
        self.last_conv = conv(n_feats, n_feats, kernel_size)

        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, num, weights_dict=None):
        if weights_dict != None:
            x = self.sub_mean(x)
            #x = self.head(x)
            x = edsr_head(x, 'model.head', weights_dict)
            

            #adafm
            if self.adafm or self.maml:
                res = x
                for i in range(self.numbers):
                    #res = self.body[i](res, num)
                    res = edsr_resblock(res, 'model.body.{}'.format(i), weights_dict)
                    
                res = edsr_conv(res, 'model.last_conv', weights_dict)
                #res = self.body[self.n_resblocks](res)
                #res = self.last_conv(res)
                res += x
                x = edsr_tail(res, 'model.tail', weights_dict)
                #x = self.tail(res)

            #original
            else:
                res = self.body(x)
                res += x
                x = self.tail(res)
            x = self.add_mean(x)
        else:
            _,_,h,w = x.shape
            
            x = self.sub_mean(x)
            #x = self.head(x)
            #adafm
            if self.adafm or self.maml:
                res = x
                for i in range(self.numbers):
                    res = self.body[i](res, num)
                #res = self.body[self.n_resblocks](res)
                res = self.last_conv(res)
                res += x

            #original
            else:
                res = self.body(x)
                #res += x

            x = self.tail(res)
            x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
            # Adafm改变模型结构，conv从2变为3
            else:
                #print(name)
                name = name.replace("2.weight","3.weight") if "weight" in name else name.replace("2.bias","3.bias")
                if isinstance(param, nn.Parameter):
                    param = param.data
                own_state[name].copy_(param)

