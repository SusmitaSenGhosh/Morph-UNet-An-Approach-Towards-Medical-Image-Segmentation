import torch.nn as nn
from torch.nn import functional as F
import torch 
import math


class Morphology(nn.Module):
    '''
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, kernel_size=5, soft_max=True, dilation = 1,type=None):
        '''
        in_channels: scalar
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        type: str, dilation2d or erosion2d or gradient2d.
        '''
        super(Morphology, self).__init__()
        # if kernel_size%2 == 0:
        #     p = int(((kernel_size-1)*(dilation-1)+kernel_size)/2)
        #     # p1 = dilation*(kernel_size-1)//2
        #     # print(kernel_size,p,p1)
        # else:
        p = int(((kernel_size-1)*(dilation-1)+kernel_size)/2)
            # p1 = dilation*(kernel_size-1)//2
            # print(kernel_size,p,p1)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.type = type
        self.dilation = dilation
        self.weight = nn.Parameter(torch.zeros(in_channels, kernel_size, kernel_size), requires_grad=True)
        # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        self.unfold = nn.Unfold(kernel_size, dilation=dilation, padding=p, stride=1)
        self.activation = nn.Softmax(dim =-1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        l1 = x.shape[-1]
        x = self.unfold(x)
        x = torch.permute(x,[0,2,1])
        x = x.view(-1, x.shape[1], self.in_channels,self.kernel_size*self.kernel_size)

        w = self.weight.view(self.in_channels,-1)
        w = w.unsqueeze(0).unsqueeze(1)


        if self.type == 'dilation2d':
            x,_= torch.max(x+w, dim=-1, keepdim=False) # (B, Cout, L)
        elif self.type == 'erosion2d':
            x,_ = torch.min(x-w, dim=-1, keepdim=False) # (B, Cout, L)
        elif self.type == 'gradient2d':
            xd,_ = torch.max(x+w, dim=-1, keepdim=False)
            xe,_ = torch.min(x-w, dim=-1, keepdim=False)
            x = xd-xe
        else:
            raise ValueError

        if self.soft_max:
            x = self.activation(x)

        x = torch.permute(x,[0,2,1])
        l = int(math.sqrt(x.shape[-1]))
        if l1 != l:
          raise Exception("Something is wrong")
        x = x.view(-1, self.in_channels, l, l)

        return x 

class Dilation2d(Morphology):
    def __init__(self, in_channels, kernel_size=5, soft_max=True,dilation = 1):
        super(Dilation2d, self).__init__(in_channels, kernel_size, soft_max, dilation, 'dilation2d')

class Erosion2d(Morphology):
    def __init__(self, in_channels, kernel_size=5, soft_max=True,dilation = 1):
        super(Erosion2d, self).__init__(in_channels, kernel_size, soft_max, dilation,'erosion2d')

class Gradient2d(Morphology):
    def __init__(self, in_channels, kernel_size=5, soft_max=True, dilation = 1):
        super(Gradient2d, self).__init__(in_channels, kernel_size, soft_max, dilation,'gradient2d')


    
class opening(nn.Module):
    def __init__(self, in_channel,kernel,soft_max=True,dilation = 1):
        super(opening, self).__init__()
        n = in_channel
        self.ers = Erosion2d(in_channel, kernel, soft_max=True,dilation = dilation)
        self.dil = Dilation2d(in_channel, kernel, soft_max=True,dilation = dilation)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        batch, _, h, w = x.size()
        x = self.bn1(self.ers(x))
        x = self.bn2(self.dil(x))
        return self.relu(x)  

class closing(nn.Module):
    def __init__(self, in_channel,kernel,soft_max=True,dilation = 1):
        super(closing, self).__init__()
        n = in_channel
        self.ers = Erosion2d(in_channel, kernel, soft_max=True,dilation = dilation)
        self.dil = Dilation2d(in_channel, kernel, soft_max=True,dilation = dilation)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        batch, _, h, w = x.size()
        x = self.bn1(self.dil(x))
        x = self.bn2(self.ers(x))
        return self.relu(x) 