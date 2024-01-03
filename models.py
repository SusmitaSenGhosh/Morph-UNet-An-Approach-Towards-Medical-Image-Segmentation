import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import torchvision.models as models
# from torchsummary import summary
# import math
from morpholayers import *
from unet_part import *



class MSDCM(nn.Module):
    def __init__(self, in_channel,kernel):
        super(MSDCM, self).__init__()
        n = in_channel

        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel, padding=int((kernel-1)*1/2), dilation=1, bias=False,groups = in_channel)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel, padding=int((kernel-1)*2/2), dilation=2, bias=False,groups = in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.conv3 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel, padding=int((kernel-1)*3/2), dilation=3, bias=False,groups = in_channel)
        self.bn3 = nn.BatchNorm2d(in_channel)
        self.conv4 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel, padding=int((kernel-1)*4/2), dilation=4, bias=False,groups = in_channel)
        self.bn4 = nn.BatchNorm2d(in_channel)


        self.conv = nn.Conv2d(n*4, n, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(n)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        batch, _, h, w = x.size()

        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.drop(self.relu(self.bn(self.conv(x))))

        return x



class MSMCM(nn.Module):
    def __init__(self, in_channel,kernel):
        super(MSMCM, self).__init__()
        n = in_channel

        self.grad1 = closing(n, kernel, soft_max=True,dilation = 1)
        self.grad2 = closing(n, kernel, soft_max=True,dilation = 2)
        self.grad3 = closing(n, kernel, soft_max=True,dilation = 3)
        self.grad4 = closing(n, kernel, soft_max=True,dilation = 4)

        self.bn1 = nn.BatchNorm2d(n)
        self.bn2 = nn.BatchNorm2d(n)
        self.bn3 = nn.BatchNorm2d(n)
        self.bn4 = nn.BatchNorm2d(n)

        self.conv = nn.Conv2d(n*4, n, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(n)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        batch, _, h, w = x.size()
        x1 = self.relu(self.bn1(self.grad1(x)))
        x2 = self.relu(self.bn2(self.grad2(x)))
        x3 = self.relu(self.bn3(self.grad3(x)))
        x4 = self.relu(self.bn4(self.grad4(x)))

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.drop(self.relu(self.bn(self.conv(x))))

        return x  

class MSMOM(nn.Module):
    def __init__(self, in_channel,kernel):
        super(MSMOM, self).__init__()
        n = in_channel

        self.grad1 = opening(n, kernel, soft_max=True,dilation = 1)
        self.grad2 = opening(n, kernel, soft_max=True,dilation = 2)
        self.grad3 = opening(n, kernel, soft_max=True,dilation = 3)
        self.grad4 = opening(n, kernel, soft_max=True,dilation = 4)

        self.bn1 = nn.BatchNorm2d(n)
        self.bn2 = nn.BatchNorm2d(n)
        self.bn3 = nn.BatchNorm2d(n)
        self.bn4 = nn.BatchNorm2d(n)

        self.conv = nn.Conv2d(n*4, n, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(n)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        batch, _, h, w = x.size()
        x1 = self.relu(self.bn1(self.grad1(x)))
        x2 = self.relu(self.bn2(self.grad2(x)))
        x3 = self.relu(self.bn3(self.grad3(x)))
        x4 = self.relu(self.bn4(self.grad4(x)))

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.drop(self.relu(self.bn(self.conv(x))))

        return x  

class MSMGM(nn.Module):
    def __init__(self, in_channel,kernel):
        super(MSMGM, self).__init__()
        n = in_channel

        self.grad1 = Gradient2d(n, kernel, soft_max=True,dilation = 1)
        self.grad2 = Gradient2d(n, kernel, soft_max=True,dilation = 2)
        self.grad3 = Gradient2d(n, kernel, soft_max=True,dilation = 3)
        self.grad4 = Gradient2d(n, kernel, soft_max=True,dilation = 4)

        self.bn1 = nn.BatchNorm2d(n)
        self.bn2 = nn.BatchNorm2d(n)
        self.bn3 = nn.BatchNorm2d(n)
        self.bn4 = nn.BatchNorm2d(n)

        self.conv = nn.Conv2d(n*4, n, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(n)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        batch, _, h, w = x.size()
        x1 = self.relu(self.bn1(self.grad1(x)))
        x2 = self.relu(self.bn2(self.grad2(x)))
        x3 = self.relu(self.bn3(self.grad3(x)))
        x4 = self.relu(self.bn4(self.grad4(x)))

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.drop(self.relu(self.bn(self.conv(x))))

        return x  


class UNet_MSMM(nn.Module):
    def __init__(self,n_channels, n_classes, bilinear=False,  module = 'MSMCM'):
        super(UNet_MSMM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        c = 3
        self.inc = (myDoubleConv(n_channels, 16))
        self.down1 = (Down(16,32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        if module == 'MSMCM':
            self.maspp = MSMCM(256,c)
        if module == 'MSMOM':
            self.maspp = MSMOM(256,c)
        if module == 'MSMGM':
            self.maspp = MSMGM(256,c)
        if module ==  'MSDCM':
            self.maspp = MSDCM(256,c)
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConvNoAF(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) 
        x5 = self.down4(x4)
        x = self.maspp(x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class UNet_MSMAM(nn.Module):
    def __init__(self,n_channels, n_classes, bilinear=False,  module = 'MSMAM'):
        super(UNet_MSMAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        c = 3
        self.inc = (myDoubleConv(n_channels, 16))
        self.down1 = (Down(16,32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        if module == 'MSMAM':
            self.masppC = MSMCM(256,c)
            self.masppO = MSMOM(256,c)
            self.masppG = MSMGM(256,c)
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConvNoAF(16, n_classes))
        self.conv = nn.Conv2d(768,256,1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) 
        x5 = self.down4(x4)
        xc = self.masppC(x5)
        xo = self.masppO(x5)
        xg = self.masppG(x5)
        x = torch.cat((xc, xo, xg), 1)
        # print(x.shape)
        x = self.conv(x)
        # print(x1.shape)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Morph_UNet_MSMM(nn.Module):
    def __init__(self,n_channels, n_classes, bilinear=False,  module = 'MSMGM'):
        super(Morph_UNet_MSMM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        c = 3
        self.inc = (myDoubleConv(n_channels, 16))
        self.down1 = (Down(16,32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        if module == 'MSMCM':
            self.maspp1 = MSMCM(256,c)
            self.maspp2 = MSMCM(128,c)
            self.maspp3 = MSMCM(64,c)
        if module == 'MSMOM':
            self.maspp1 = MSMOM(256,c)
            self.maspp2 = MSMOM(128,c)
            self.maspp3 = MSMOM(64,c)
        if module == 'MSMGM':
            self.maspp1 = MSMGM(256,c)
            self.maspp2 = MSMGM(128,c)
            self.maspp3 = MSMGM(64,c)
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConvNoAF(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3_skip = self.maspp3(x3)
        x4 = self.down3(x3) 
        x4_skip = self.maspp2(x4)
        x5 = self.down4(x4)
        x = self.maspp1(x5)
        x = self.up1(x, x4_skip)
        x = self.up2(x, x3_skip)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_vanila(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_vanila, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 16))
        self.down1 = (Down(16, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down4 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 16, bilinear))
        self.outc = (OutConvNoAF(16, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits





class Morpho_UNet_ResNet34(nn.Module):
    def __init__(self, backbone = 'ResNet34',GrdaModule = 'mspp', pretrained=True, n_classes=1,bilinear = False, module= 'MSMGM'):
        super(Morpho_UNet_ResNet34, self).__init__()
        self.encoder = resnet34bottom(backbone,pretrained)
        if backbone == 'ResNet34':
            channels = [64,128,256]
        # channels = [256,512,1024]
        if module == 'MSMGM':
            self.module1  = MSMGM(channels[0],3)
            self.module2  = MSMGM(channels[1],3)
            self.module3  = MSMGM(channels[2],3)
        elif module == 'MSMCM':
            self.module1  = MSMCM(channels[0],3)
            self.module2  = MSMCM(channels[1],3)
            self.module3  = MSMCM(channels[2],3)            
        elif module == 'MSMOM':
            self.module1  = MSMOM(channels[0],3)
            self.module2  = MSMOM(channels[1],3)
            self.module3  = MSMOM(channels[2],3)

        self.up1 = myUp(channels[-1],channels[-2], bilinear)
        self.up2 = myUp(channels[-2],channels[-3], bilinear)
        # self.up3 = (Up(64, 32, bilinear))

        # self.outc = (OutConvNoAF(channels[-3], n_classes))
        # self.outc1 = (OutConvNoAF(channels[-2], n_classes))
        # self.outc2 = (OutConvNoAF(channels[-1], n_classes))

        self.outc = (OutConvNoAF(channels[-3], n_classes))
        self.outc1 = (OutConvNoAF(channels[-2], n_classes))
        self.outc2 = (OutConvNoAF(channels[-1], n_classes))


    def forward(self, x):
        H= x.shape[-2]
        W = x.shape[-1]
        x3,x2,x1 = self.encoder(x)
        # print(x3.shape,x2.shape,x1.shape)
        x1_skip = self.module1(x1)
        x2_skip = self.module2(x2)
        x = self.module3(x3)        
        x22 = x
        x = self.up1(x, x2_skip)
        x11 = x
        x = self.up2(x, x1_skip)
        logits = self.outc(x)
        logits1 = self.outc1(x11)
        logits2 = self.outc2(x22)

        map_x = F.interpolate(logits, scale_factor=int(256/logits.shape[-1]), mode='bilinear', align_corners=True)
        map_1 = F.interpolate(logits1, scale_factor=int(256/logits1.shape[-1]), mode='bilinear', align_corners=True)
        map_2 = F.interpolate(logits2, scale_factor=int(256/logits2.shape[-1]), mode='bilinear', align_corners=True)  

        return map_x#, map_1, map_2
        
class Morpho_ResNet34_decoder(nn.Module):
    def __init__(self, backbone = 'ResNet34',GrdaModule = 'mspp', pretrained=True, n_classes=1,bilinear = False, module= 'MSMGM'):
        super(Morpho_ResNet34_decoder, self).__init__()
        self.encoder = resnet34bottom(backbone,pretrained)
        if backbone == 'ResNet34':
            channels = [64,128,256]
        # channels = [256,512,1024]
        if module == 'MSMGM':
            self.module1  = MSMGM(channels[0],3)
            self.module2  = MSMGM(channels[1],3)
            #self.module3  = MSMGM(channels[2],3)
        elif module == 'MSMCM':
            self.module1  = MSMCM(channels[0],3)
            self.module2  = MSMCM(channels[1],3)
            #self.module3  = MSMCM(channels[2],3)            
        elif module == 'MSMOM':
            self.module1  = MSMOM(channels[0],3)
            self.module2  = MSMOM(channels[1],3)
            #self.module3  = MSMOM(channels[2],3)

        self.up1 = myUp(channels[-1],channels[-2], bilinear)
        self.up2 = myUp(channels[-2],channels[-3], bilinear)
        # self.up3 = (Up(64, 32, bilinear))

        # self.outc = (OutConvNoAF(channels[-3], n_classes))
        # self.outc1 = (OutConvNoAF(channels[-2], n_classes))
        # self.outc2 = (OutConvNoAF(channels[-1], n_classes))

        self.outc = (OutConvNoAF(channels[-3], n_classes))
        #self.outc1 = (OutConvNoAF(channels[-2], n_classes))
        #self.outc2 = (OutConvNoAF(channels[-1], n_classes))


    def forward(self, x):
        H= x.shape[-2]
        W = x.shape[-1]
        x3,x2,x1 = self.encoder(x)
        x1_skip = x1
        x2_skip = x2
        x = self.up1(x3, x2_skip)
        x = self.module2(x)       
        x = self.up2(x, x1_skip)
        x = self.module1(x)       
        logits = self.outc(x)


        map_x = F.interpolate(logits, scale_factor=int(256/logits.shape[-1]), mode='bilinear', align_corners=True)

        return map_x

        
class resnet34bottom(nn.Module):
    def __init__(self, model_name, pretrained):
        super(resnet34bottom, self).__init__()
   
        if model_name == 'ResNet34':
          original_model = models.resnet34(pretrained=pretrained)
          
       # print(list(list(original_model.children())[0].children())[0:2])
        # self.features1 = nn.Sequential(*list(original_model.children())[:3])
        # self.features2 = nn.Sequential(*list(original_model.children())[:5])
        # self.features3 = nn.Sequential(*list(original_model.children())[:6])
        # self.features4 = nn.Sequential(*list(original_model.children())[:7])

        self.features1 = nn.Sequential(*list(original_model.children())[0:5])
        self.features2 = nn.Sequential(*list(original_model.children())[0:6])
        self.features3 = nn.Sequential(*list(original_model.children())[0:7])
    def forward(self, x):
        # x1 = self.features1(x)
        x1 = self.features1(x)
        x2 = self.features2(x)
        x3 = self.features3(x)

        #print('shapes',x1.shape,x2.shape,x3.shape)
        return x3,x2,x1




class Morpho_UNet_efficientnetb4(nn.Module):
    def __init__(self, backbone = 'efficientnet_b4', pretrained=True, n_classes=1,bilinear = False, module = 'MSMCM'):
        super(Morpho_UNet_efficientnetb4, self).__init__()
        self.encoder = effiecientnetb4bottom(backbone,pretrained)
        c = 3
        if backbone == 'efficientnet_b4':
            channels = [24,32,56]
        if module == 'MSMGM':
            self.module1  = MSMGM(channels[0],c)
            self.module2  = MSMGM(channels[1],c)
            self.module3  = MSMGM(channels[2],c)
        elif module == 'MSMCM':
            self.module1  = MSMCM(channels[0],c)
            self.module2  = MSMCM(channels[1],c)
            self.module3  = MSMCM(channels[2],c)            
        elif module == 'MSMOM':
            self.module1  = MSMOM(channels[0],c)
            self.module2  = MSMOM(channels[1],c)
            self.module3  = MSMOM(channels[2],c)
        # elif module == 'MSMCM_res':
        #     self.module1  = MSMCM_res(channels[0],3)
        #     self.module2  = MSMCM_res(channels[1],3)
        #     self.module3  = MSMCM_res(channels[2],3)

        self.up1 = myUp(channels[-1],channels[-2], bilinear)
        self.up2 = myUp(channels[-2],channels[-3], bilinear)

        self.outc = (OutConvNoAF(channels[-3], n_classes))
        self.outc1 = (OutConvNoAF(channels[-2], n_classes))
        self.outc2 = (OutConvNoAF(channels[-1], n_classes))

    def forward(self, x):
        H= x.shape[-2]
        W = x.shape[-1]
        x3,x2,x1 = self.encoder(x)
        x1_skip = self.module1(x1)
        x2_skip = self.module2(x2)
        x = self.module3(x3)        
        x22 = x
        x = self.up1(x, x2_skip)
        x11 = x
        x = self.up2(x, x1_skip)
        logits = self.outc(x)
        logits1 = self.outc1(x11)
        logits2 = self.outc2(x22)

        map_x = F.interpolate(logits, scale_factor=int(256/logits.shape[-1]), mode='bilinear', align_corners=True)
        map_1 = F.interpolate(logits1, scale_factor=int(256/logits1.shape[-1]), mode='bilinear', align_corners=True)
        map_2 = F.interpolate(logits2, scale_factor=int(256/logits2.shape[-1]), mode='bilinear', align_corners=True)  

        return map_x#, map_1, map_2


class effiecientnetb4bottom(nn.Module):
    def __init__(self, model_name, pretrained):
        super(effiecientnetb4bottom, self).__init__()
   
        if model_name == 'efficientnet_b4':
          original_model = models.efficientnet_b4(pretrained=pretrained)
          
        # print(list(list(original_model.children())[0].children())[0:2])
        # self.features1 = nn.Sequential(*list(original_model.children())[:3])
        # self.features2 = nn.Sequential(*list(original_model.children())[:5])
        # self.features3 = nn.Sequential(*list(original_model.children())[:6])
        # self.features4 = nn.Sequential(*list(original_model.children())[:7])

        self.features1 = nn.Sequential(*list(list(original_model.children())[0].children())[0:2])
        self.features2 = nn.Sequential(*list(list(original_model.children())[0].children())[0:3])
        self.features3 = nn.Sequential(*list(list(original_model.children())[0].children())[0:4])
    def forward(self, x):
        # x1 = self.features1(x)
        x1 = self.features1(x)
        x2 = self.features2(x)
        x3 = self.features3(x)

        # print(x1.shape,x2.shape,x3.shape)
        return x3,x2,x1


