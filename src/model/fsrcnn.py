import torch
import torch.nn as nn
import math
import torch.nn.init as init
from model import common
import torch.nn.functional as F

"""Image Super-Resolution Using Deep Convolutional Networks"""


def make_model(args, parent=False):
    return FSRCNN(args)
    # return SRCNNCond(args)

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def fsrcnn_prelu(input, prefix,weights_dict):
    x = F.prelu(input, weights_dict[prefix + '.weight'])
    return x

def fsrcnn_conv(input, prefix, weights_dict,padding):
    #print( 'input',input.shape,'conv',weights_dict[prefix + '.weight'].shape)
    x = F.conv2d(input, weights_dict[prefix + '.weight'], weights_dict[prefix + '.bias'],padding=padding)
    return x

def fsrcnn_transconv(input, prefix, weights_dict,scale):
    x = F.conv_transpose2d(input,weights_dict[prefix + '.weight'], weights_dict[prefix + '.bias'],stride=scale, padding=9//2, output_padding=scale-1)
    return x





class FSRCNN(nn.Module):
    def __init__(self, args):
        super(FSRCNN, self).__init__()
        self.scale = int(args.scale[0])
        self.n_colors = args.n_colors
        
        
        self.conv1 = nn.Conv2d(self.n_colors, 56, kernel_size=5, padding=5//2)
        self.PReLU1 = nn.PReLU(56)
        self.conv2 = nn.Conv2d(56, 12, kernel_size=1)
        self.PReLU2 = nn.PReLU(12)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=3//2)
        self.PReLU3 = nn.PReLU(12)
        self.conv4 = nn.Conv2d(12, 12, kernel_size=3, padding=3//2)
        self.PReLU4 = nn.PReLU(12)
        self.conv5 = nn.Conv2d(12, 12, kernel_size=3, padding=3//2)
        self.PReLU5 = nn.PReLU(12)
        self.conv6 = nn.Conv2d(12, 12, kernel_size=3, padding=3//2)
        self.PReLU6 = nn.PReLU(12)
        self.conv7 = nn.Conv2d(12, 56, kernel_size=1)
        self.PReLU7 = nn.PReLU(56)
        self.transconv1 =  nn.ConvTranspose2d(56, 3, kernel_size=9, stride=self.scale, padding=9//2,
                                            output_padding=self.scale-1)


        self._initialize_weights()

    def _initialize_weights(self):

        for m in range(1,8):
            varConv = 'self.conv{}'.format(m)
            nn.init.normal_(eval(varConv).weight.data, mean=0.0, std=math.sqrt(2/(eval(varConv).out_channels*eval(varConv).weight.data[0][0].numel())))
            nn.init.zeros_(eval(varConv).bias.data)
            
            
        nn.init.normal_(self.transconv1.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.transconv1.bias.data)



    def forward(self, x,num, weights_dict=None):
        if weights_dict != None:
            #print('11',x.shape)
            
            x = fsrcnn_conv(x,'model.conv1',weights_dict,padding=5//2)
            #print('12_conv',x.shape)
            x = fsrcnn_prelu(x,'model.PReLU1',weights_dict)
            #print('12',x.shape)
            x = fsrcnn_prelu(fsrcnn_conv(x,'model.conv2',weights_dict,padding=0),'model.PReLU2',weights_dict)
            #print('13',x.shape)
            for i in range(3,7):
                x = fsrcnn_prelu(fsrcnn_conv(x,'model.conv{}'.format(i),weights_dict,padding=3//2),'model.PReLU{}'.format(i),weights_dict)
                #print('14',x.shape)
            x = fsrcnn_prelu(fsrcnn_conv(x,'model.conv7',weights_dict,padding=0),'model.PReLU7',weights_dict)
            #print('15',x.shape)
            x = fsrcnn_transconv(x,'model.transconv1',weights_dict,self.scale)
            #print('16',x.shape)
            return x
        else:
            #print('21',x.shape)
            for i in range(1,8):
                varConv = 'self.conv{}'.format(i)
                
                varPreLU = 'self.PReLU{}'.format(i)
                x = eval(varConv)(x)
                #print('22_conv',x.shape)
                x = eval(varPreLU)(x)
                #print('22',x.shape)
            
            x = self.transconv1(x)
            #print('23',x.shape)
            return x

# class SRCNNCond(nn.Module):
#     def __init__(self, args):
#         super(SRCNNCond, self).__init__()
#         n_colors = args.n_colors
#         n = 4
#         self.conv1 = common.CondConv2d(in_channels=n_colors, out_channels=64, kernel_size=9, stride=1, padding=9//2,
#                                        num=n)
#         self.conv2 = common.CondConv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=5//2, num=n)
#         self.conv3 = common.CondConv2d(in_channels=32, out_channels=n_colors, kernel_size=5, stride=1, padding=5//2,
#                                        num=n)
#         self.relu = nn.ReLU(inplace=True)
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         self.conv1.initialize_weights(init.calculate_gain('relu'))
#         self.conv2.initialize_weights(init.calculate_gain('relu'))
#         self.conv3.initialize_weights(1)

#     def forward(self, x, krl=None,weights_dict=None):

#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x)
#         return x




if __name__ == '__main__':
    print(__file__)