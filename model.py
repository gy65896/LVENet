import torch
import torch.nn as nn
import torch.nn.functional as F


class conv(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size=3,stride=1,padding=1):
        super(conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class dconv(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size=3,stride=1,padding=1):
        super(dconv, self).__init__()
        self.depth_conv = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch)
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1)
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class Resblock(nn.Module):
    def __init__(self, out_c,kernel):
        super(Resblock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.InstanceNorm2d(out_c, affine=True) 
        self.conv1 = conv(out_c,out_c, kernel_size=kernel,padding=kernel//2)
    
    def forward(self,x):
        
        x_res = x
        x_res = self.relu(self.conv1(x_res))
        x_res = self.conv1(x_res)
        
        x = x + x_res
        return x

class IEM(nn.Module):
    def __init__(self):
        super(IEM,self).__init__()
        self.conv_3_3   = conv(3 ,8 ,kernel_size=3,stride=1,padding=0)
        self.conv_3_6   = conv(8 ,16,kernel_size=3,stride=1,padding=0)
        self.conv_6_12  = conv(16,32,kernel_size=3,stride=1,padding=0)
        self.conv_12_12_1 = Resblock(32,3)
        self.conv_12_12_2 = Resblock(32,3)
        self.conv_12_12_3 = Resblock(32,3)
        self.conv_12_6  = dconv(32,16,kernel_size=3,stride=1,padding=0)
        self.conv_6_3   = dconv(16,8 ,kernel_size=3,stride=1,padding=0)
        self.conv_3_1   = dconv(8 ,1 ,kernel_size=3,stride=1,padding=0)
        self.relu = nn.ReLU(inplace=True)
    def _upsample(self,x,y):
        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear')
    def forward(self,x):
        print(x.shape)
        conv1 = self.relu(self.conv_3_3(x))
        print(conv1.shape)
        conv2 = self.relu(self.conv_3_6(conv1))
        print(conv2.shape)
        conv3 = self.relu(self.conv_6_12(conv2))
        print(conv3.shape)
        conv3 = self.relu(self.conv_12_12_1(conv3))
        conv3 = self.relu(self.conv_12_12_2(conv3))
        print(conv3.shape)
        conv3 = self.relu(self.conv_12_12_3(conv3))
        conv4 = self.relu(self.conv_12_6(conv3))
        print(conv4.shape)
        conv5 = self.relu(self.conv_6_3(conv4))
        conv6 = self.relu(self.conv_3_1(conv5))
        
        conv8 = torch.clamp(conv6,0.05,1)
        L_out = torch.cat((conv8,conv8,conv8),1)
        R_out=torch.clamp(x*(conv8**(-1)),0,1)
        
        return R_out, L_out