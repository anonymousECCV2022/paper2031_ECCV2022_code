import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
#from .sync_batchnorm import BatchNorm1d, BatchNorm2d

BatchNorm1d = nn.BatchNorm1d
BatchNorm2d = nn.BatchNorm2d

__all__ = ["DGR"]

class DGR(nn.Module):
    def __init__(self, in_channels, K):
        super(DGR, self).__init__()
        self.project = A2_Projection(in_channels, K)
        self.gcn = GraphConvolution(in_channels, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.bn = BatchNorm2d(in_channels, eps=1e-4)
        nn.init.constant_(self.bn.weight, 0)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        B, C, H, W = x.size()
        graph, q = self.project(x)
        graph = self.gcn(graph) * self.gamma + graph
        output = torch.matmul(graph, q.permute(0, 2, 1)).reshape(B, C, H, W)
        output = self.bn(output) + x
        return output

class A2_Projection(nn.Module):
    def __init__(self, in_channels, K):
        super(A2_Projection, self).__init__()
        self.in_channels = in_channels
        self.num_state = in_channels #256
        self.K = K
        self.phi = nn.Sequential(
            nn.Conv2d(self.in_channels, self.K, 1, bias=False),
            BatchNorm2d(self.K),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_state, 1, bias=False),
            BatchNorm2d(self.num_state),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.rou = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_state, 1, bias=False),
            BatchNorm2d(self.num_state),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_state, 1, bias=False),
            BatchNorm2d(self.num_state),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
            X: B, 256, H, W
        """
        assert x.dim() == 4

        # BxDxHxW
        B, C, H, W = x.size()

        # Stage 1: generate discribtors
        x_phi = self.phi(x).view(B, -1, H*W) # B, K, N
        x_theta = self.theta(x).view(B, -1, H*W) # B, C, N
        x_theta = self.softmax(x_theta).permute(0, 2, 1) # B, N, C
        discrib = torch.matmul(x_phi, x_theta) # B, K, C

        # Stage2: Encode features
        x_rou = self.rou(x).view(B, -1, H*W) # B, C, N
        x_rou = self.softmax(x_rou.permute(0,2,1)).permute(0,2,1) # B, C, N
        Q = torch.matmul(discrib, x_rou).permute(0, 2, 1) # B, N, K

        # Stage3: extract similar regions by Q
        Q = F.normalize(Q, p=2, dim=1)
        Q = self.softmax(Q)
        x = self.value(x).reshape(B, self.num_state, H*W).unsqueeze(-1) # B, D, N, 1 
        Z = (((Q.unsqueeze(1) * x).sum(2))/Q.sum(1).unsqueeze(1)) #B, D, N, K -> B, D, K -> B, D, K /B, K -> B, D, K
        Z = F.normalize(Z, p=2, dim=1) #B, D, K
        # Z = F.dropout(Z, p=0.05)

        return Z, Q

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    *** Cosine similarity adjcent matrix
    """
    def __init__(self, in_features, out_features, norm_layer=BatchNorm1d, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = norm_layer(out_features)
        nn.init.constant_(self.bn.weight, 0)
        nn.init.constant_(self.bn.bias, 0)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def self_similarity(self, x):
        '''
        input: B, n_class, n_channel
        out: B, n_class, n_class
        '''
        sim = torch.matmul(x, x.transpose(-1, -2))
        sim = F.softmax(sim, dim=-1) #-1指的是方向
        return sim

    def forward(self, input):
        input = input.permute(0, 2, 1)
        adj = self.softmax(torch.matmul(input, input.transpose(-1, -2)))
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support).permute(0, 2, 1)

        if self.bias is not None:
            return self.bn(self.relu(output + self.bias)) 
        else:
            return self.bn(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
