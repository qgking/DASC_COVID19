import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from module.modules import BaseModule
import torch.nn.init as init


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_normal_2d(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 70*70   patchgan2
class simple_discriminator(nn.Module):
    # initializers
    def __init__(self, in_channels=1, filters=64):
        super(simple_discriminator, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, filters, 4, 2, 1)
        self.conv2 = nn.Conv3d(filters, filters * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm3d(filters * 2)
        self.conv3 = nn.Conv3d(filters * 2, filters * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d(filters * 4)
        self.conv4 = nn.Conv3d(filters * 4, filters * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm3d(filters * 8)
        self.conv5 = nn.Conv3d(filters * 8, 1, 4, 1, 1)
        self.apply(weights_init_normal)

    # forward method
    def forward(self, input):
        x = input
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x


class SNTemporalPatchGANDiscriminator(BaseModule):
    def __init__(
            self, nc_in=2, nf=64, norm='SN', use_sigmoid=False, use_bias=True, conv_type='vanilla',
            conv_by='3d'
    ):
        super().__init__(conv_type)
        use_bias = use_bias
        self.use_sigmoid = use_sigmoid

        ######################
        # Convolution layers #
        ######################
        # self.conv1 = self.ConvBlock(
        #     nc_in, nf * 1, kernel_size=(5, 5, 5), stride=(2, 2, 2),
        #     padding=1, bias=use_bias, norm=norm, conv_by=conv_by
        # )
        # self.conv2 = self.ConvBlock(
        #     nf * 1, nf * 2, kernel_size=(5, 5, 5), stride=(2, 2, 2),
        #     padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        # )
        # self.conv3 = self.ConvBlock(
        #     nf * 2, nf * 4, kernel_size=(5, 5, 5), stride=(2, 2, 2),
        #     padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        # )
        # self.conv4 = self.ConvBlock(
        #     nf * 4, nf // nf, kernel_size=(5, 5, 5), stride=(2, 2, 2),
        #     padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        # )
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(3, 3, 3), stride=(2, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by
        )
        # receptive field 1+(3-1)=3
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(3, 3, 3), stride=(2, 2, 2),
            padding=(1, 1, 1), bias=use_bias, norm=norm, conv_by=conv_by
        )
        # receptive field 3+(3-1)*2=7
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(3, 3, 3), stride=(2, 2, 2),
            padding=(1, 1, 1), bias=use_bias, norm=norm, conv_by=conv_by
        )
        # receptive field 7+(3-1)*2*2=15
        self.conv4 = self.ConvBlock(
            nf * 4, nf // nf, kernel_size=(3, 3, 3), stride=(2, 2, 2),
            padding=(1, 1, 1), bias=use_bias, norm=norm, conv_by=conv_by
        )
        # receptive field 15+(3-1)*2*2*2=31
        # self.conv5 = self.ConvBlock(
        #     nf * 4, nf * 4, kernel_size=(5, 5, 5), stride=(2, 2, 2),
        #     padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        # )
        # self.conv6 = self.ConvBlock(
        #     nf * 4, nf // nf, kernel_size=(5, 5, 5), stride=(2, 2, 2),
        #     padding=(2, 2, 2), bias=use_bias, norm=None, activation=None,
        #     conv_by=conv_by
        # )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        c1 = self.conv1(xs)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        # c5 = self.conv5(c4)
        # c6 = self.conv6(c5)
        if self.use_sigmoid:
            c4 = torch.sigmoid(c4)
        return c4


class latent_discriminator(nn.Module):
    # initializers
    def __init__(self, in_channels=1, filters=64, backbone='resnet34'):
        super(latent_discriminator, self).__init__()
        if int(backbone[6:]) > 34:
            dim = 2048
        else:
            dim = 512
        # self.conv1 = nn.Conv2d(in_channels, 256, 4, 2, 1)
        # self.conv2 = nn.Conv2d(256, 128, 4, 2, 1)
        # self.conv3 = nn.Conv2d(128, 64, 4, 2, 1)
        # self.conv4 = nn.Conv2d(64, 32, 4, 2, 1)
        # self.conv5 = nn.Conv2d(32, 1, 4, 2, 1)

        self.conv1 = nn.Conv2d(in_channels, dim // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(dim // 2, dim // 4, 4, 2, 1)
        self.conv3 = nn.Conv2d(dim // 4, dim // 8, 4, 2, 1)
        self.conv4 = nn.Conv2d(dim // 8, 64, 4, 2, 1)
        self.conv5 = nn.Conv2d(64, 1, 4, 2, 1)
        self.apply(weights_init_normal)

    # forward method
    def forward(self, input_su):
        x = F.leaky_relu(self.conv1(input_su), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)
        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self, in_channels=1, filters=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 4, 2, 1)
        self.conv2 = nn.Conv2d(filters, filters * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(filters * 2, filters * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(filters * 4, filters * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(filters * 8, 1, 4, 2, 1)
        self.apply(weights_init_normal)

    # forward method
    def forward(self, input_su):
        x = F.leaky_relu(self.conv1(input_su), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)
        return x
