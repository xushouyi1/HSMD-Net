import torch
from torch import nn




import torch
import torch.nn as nn
from innovatory_test2 import *
import torch.nn.functional as F
# from dcn import  DeformableConv2d
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    # 初始化调用block （1024，reflect，nn.ReLU(True)，instance_norm）
    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # 通道不变卷积  padding=0， 尺寸不变
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        # 这个没有用到
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        # 又来了一次卷积
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # 图像经过两次卷积并且在加上原图 ，更换FFA在这个模块
        out = x + self.conv_block(x)
        return out




class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y



class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res



class Base_Model(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect', n_blocks=6):
        super(Base_Model,self).__init__()

        # 下采样
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.InstanceNorm2d(ngf),
                                   nn.ReLU(True))

        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(ngf*2),
                                   nn.ReLU(True))

        self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(ngf * 4),
                                   nn.ReLU(True))

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU(True)
        model_res = []
        for i in range(n_blocks):
            model_res += [ResnetBlock(ngf * 4, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.model_res = nn.Sequential(*model_res)

        # 上采样
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf*2),
                                 nn.ReLU(True))


        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf),
                                 nn.ReLU(True))

        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

        # 用于transform的下采样
        self.down2_1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=5, stride=4, padding=2, dilation=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.down2_2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.down2_3 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        # 用于transform的上采样
        self.up2_3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=3, padding=1,dilation=1,output_padding=3,stride=4),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        self.up2_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=2, output_padding=1,dilation=2), # 通道256到128 尺寸64
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True))
        self.up2_1 = nn.Sequential(
            # nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=2, output_padding=0,dilation=2),  # 通道64到256 ，尺寸 64到64
            nn.Conv2d(256, 256, kernel_size=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True))

        self.pa1 = PALayer(256)
        self.pa2 = PALayer(128)
        self.pa3 = PALayer(64)

        self.ca1 = CALayer(256)
        self.ca2 = CALayer(128)
        self.ca3 = CALayer(64)

        self.CGR1 = MDCRM(128, 256)
        self.CGR2 = MDCRM(64, 128)
        self.tf = HIISM(256)
        # self.tf2 = test_jh()




    def forward(self, input):


        x_down1 = self.down1(input)  # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1)  # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2)  # [bs, 256, 64, 64]

        # x_down1 从64转到256
        x1 = self.down2_1(x_down1)
        # x_down2 从128转到256
        x2 = self.down2_2(x_down2)
        # x_down3 从256转到256
        x3 = self.down2_3(x_down3)

        #x1_2 = self.tf1(x1, x2)
        x1_2_3 = self.tf(x1, x2, x3)

        rx3 = self.up2_1(x1_2_3)  # 256
        rx2 = self.up2_2(x1_2_3)  # 128
        rx1 = self.up2_3(x1_2_3)  # 64



        x6 = self.model_res(x_down3)
        x6 = self.ca1(x6)
        x6 = self.pa1(x6)

        x6_1 = x6+rx3+x_down3
        x_up1 = self.up1(x6_1)
        x_up1 = self.ca2(x_up1)
        x_up1 = self.pa2(x_up1)
        
        x6_2 = x_up1+rx2+x_down2
        x_up2 = self.up2(x6_2)
        x_up2 = self.ca3(x_up2)
        x_up2 = self.pa3(x_up2)
        
        x6_3 = x_up2+rx1+x_down1
        # x_sum = self.fu(x6_1, x6_2, x6_3)
        x_sum = self.CGR1(x6_1,x6_2)
        x_sum = self.CGR2(x_sum,x6_3)

        x_up3 = self.up3(x_sum)


        return x_up3





class Discriminator(nn.Module):
    def __init__(self, bn=False, ngf=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf * 16, 1, kernel_size=1)

        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

# class UpSample(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UpSample, self).__init__()
#         self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                                 nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
#
#     def forward(self, x):
#         x = self.up(x)
#         return x
if __name__ == '__main__':
    basemodel = Base_Model(3,3)
    a = torch.randn(4,3,512,512)
    b = basemodel(a)