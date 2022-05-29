from model import common
import torch.nn.functional as F
import torch.nn as nn

def make_model(args, parent=False):
    return PatchNet(args)

def patchnet_conv(input,prefix, weights_dict,padding):
    x = F.conv2d(input, weights_dict[prefix + '.weight'], weights_dict[prefix + '.bias'], stride=1, padding=padding)
    return x
def patchnet_bn(input,out_channels):
    bn = nn.BatchNorm2d(out_channels).cuda()
    x = bn(input)
    return x
def patchnet_act(input):
    x = F.relu(input)
    return x
def patchnet_avgPool(input):
    x = F.avg_pool2d(input,1)
    return x

def patchnet_adaptiveavgPool(input):
    x = F.adaptive_avg_pool2d(input,1)
    return x

class PatchNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(PatchNet, self).__init__()

        act = nn.LeakyReLU(0.2, True)
        in_channels = args.n_colors
        out_channels= None

        # define body module
        m_body = []
        # self.layers = ['64*1','64*1','128*1','128*1','256*1','256*1']
        self.layers = ['64*1']
        for l in self.layers:
            out_channels, number = [int(i) for i in l.split('*')]
            for i in range(number):
                m_body.append(common.Bottleneck(in_channels, out_channels, act=act,bn=True))
            in_channels = out_channels
            m_body.append(nn.AvgPool2d(2, stride=2))

        # define tail module
        m_tail = [
            nn.Conv2d(in_channels, 1, 1, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        ]

        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x,weights_dict=None):
        if weights_dict!=None:
            n = 0
            for l in self.layers:
                # print('n',n)
                # print(x.shape)
                res = x
                out_channels, number = [int(i) for i in l.split('*')]
                for i in range(number):
                    for j in range(3):
                        if j!=1:
                            res = patchnet_conv(res,'body.'+str(n)+'.res.'+str(j*3), weights_dict,padding=0)
                        else:
                            res = patchnet_conv(res,'body.'+str(n)+'.res.'+str(j*3), weights_dict,padding=1)
                        
                        if j!=2:
                            res = patchnet_bn(res,out_channels//4)
                            res = patchnet_act(res)
                        else:
                            res = patchnet_bn(res,out_channels)
                    shortcut = patchnet_conv(x,'body.'+str(n)+'.shortcut.0', weights_dict,padding=0)
                    shortcut = patchnet_bn(shortcut,out_channels)
                    x = res + shortcut
                    x = patchnet_avgPool(x)
                    n = n+2
            #tail
            x = patchnet_conv(x,'tail.0', weights_dict,padding=0)
            x = patchnet_adaptiveavgPool(x)
            Sig = nn.Sigmoid()
            x = Sig(x)
        else:
            x = self.body(x)
            #print(x.shape)
            x = self.tail(x)
            #print(x.shape)

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