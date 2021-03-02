import torch
import torch.nn as nn
import torch.nn.functional as F
from module.blocks import (
    GatedConv, GatedDeconv,
    VanillaConv, VanillaDeconv
)


###########################
# Encoder/Decoder Modules #
###########################
def crop(variable, th, tw, td):
    h, w, d = variable.shape[2], variable.shape[3], variable.shape[4]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    z1 = int(round((d - td) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw, z1: z1 + td]


class BaseModule(nn.Module):
    def __init__(self, conv_type):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == 'gated':
            self.ConvBlock = GatedConv
            self.DeconvBlock = GatedDeconv
        elif conv_type == 'vanilla' or conv_type == 'nodilate':
            self.ConvBlock = VanillaConv
            self.DeconvBlock = VanillaDeconv


class DownSampleNormalModule(BaseModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(conv_type)
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(5, 5, 5), stride=1,
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Downsample 1
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(4, 4, 4), stride=(2, 2, 2),
            padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        # Downsample 2
        self.conv4 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(4, 4, 4), stride=(2, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # No Dilated Convolutions
        self.normal_conv1 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.normal_conv2 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.normal_conv3 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.normal_conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv7 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv8 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)

        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        a1 = self.normal_conv1(c6)
        a2 = self.normal_conv2(a1)
        a3 = self.normal_conv3(a2)
        a4 = self.normal_conv4(a3)

        c7 = self.conv7(a4)
        c8 = self.conv8(c7)
        return c8, c7, c4, c2  # For skip connection


class DownSampleModule(BaseModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(conv_type)
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(5, 5, 5), stride=1,
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Downsample 1
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(4, 4, 4), stride=(2, 2, 2),
            padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        # Downsample 2
        self.conv4 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(4, 4, 4), stride=(2, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Dilated Convolutions
        self.dilated_conv1 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(2, 2, 2))
        self.dilated_conv2 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(4, 4, 4))
        # self.dilated_conv3 = self.ConvBlock(
        #     nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
        #     padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(8, 8, 8))
        # self.dilated_conv4 = self.ConvBlock(
        #     nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
        #     padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(16, 16, 16))
        self.conv7 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv8 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)

        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        c5 = self.conv5(c4)
        c6 = self.conv6(c5)

        a1 = self.dilated_conv1(c6)
        a2 = self.dilated_conv2(a1)
        # a3 = self.dilated_conv3(a2)
        # a4 = self.dilated_conv4(a3)

        c7 = self.conv7(a2)
        c8 = self.conv8(c7)
        return c8, c7, c4, c2  # For skip connection


class AttentionDownSampleModule(DownSampleModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(nc_in, nf, use_bias, norm, conv_by, conv_type)


class UpSampleModule(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(conv_type)
        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nc_in, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        # Upsample 2
        # self.deconv2 = self.DeconvBlock(
        #     nf * 2, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
        #     bias=use_bias, norm=norm, conv_by=conv_by)

        self.deconv2 = self.DeconvBlock(
            nf * 4, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        self.conv10 = self.ConvBlock(
            nf * 1, nf * 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv11 = self.ConvBlock(
            nf * 1, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)

        self.score_dsn1 = nn.Conv3d(nf * 4, 1, kernel_size=1)
        self.score_dsn2 = nn.Conv3d(nf * 2, 1, kernel_size=1)
        self.score_dsn3 = nn.Conv3d(nf * 1, 1, kernel_size=1)
        self.score_final = nn.Conv3d(3, 1, 1)

    def concat_feature(self, ca, cb):
        if self.conv_type == 'partial':
            ca_feature, ca_mask = ca
            cb_feature, cb_mask = cb
            feature_cat = torch.cat((ca_feature, cb_feature), 1)
            # leave only the later mask
            return feature_cat, ca_mask
        else:
            return torch.cat((ca, cb), 1)

    def forward(self, inp):
        c8, c7, c4, c2 = inp
        # c8, c7, c4, c2, input = inp
        # img_H, img_W, img_D = input.shape[2], input.shape[3], input.shape[4]
        concat1 = self.concat_feature(c8, c4)
        d1 = self.deconv1(concat1)
        c9 = self.conv9(d1)
        # retrain for this part beacause its missing
        concat2 = self.concat_feature(c9, c2)
        d2 = self.deconv2(concat2)
        # d2 = self.deconv2(c9)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)

        so1_out = self.score_dsn1(c7 + c8)
        so2_out = self.score_dsn2(d1 + c9)
        so3_out = self.score_dsn3(d2 + c10)

        upsample1 = F.interpolate(so1_out, scale_factor=4, mode="trilinear", align_corners=True)
        upsample2 = F.interpolate(so2_out, scale_factor=2, mode="trilinear", align_corners=True)

        # so3_out = crop(so3_out, img_H, img_W, img_D)
        # upsample2 = crop(upsample2, img_H, img_W, img_D)
        # upsample1 = crop(upsample1, img_H, img_W, img_D)
        fusecat = torch.cat((so3_out, upsample2, upsample1), dim=1)
        fuse = self.score_final(fusecat)
        so_out_final = [so3_out, upsample2, upsample1, fuse]
        # so_out_final = [torch.sigmoid(r) for r in so_out_final]
        return c11, so_out_final


class NoRicherUpSampleModule(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(conv_type)
        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nc_in, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        # Upsample 2
        # self.deconv2 = self.DeconvBlock(
        #     nf * 2, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
        #     bias=use_bias, norm=norm, conv_by=conv_by)

        self.deconv2 = self.DeconvBlock(
            nf * 4, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        self.conv10 = self.ConvBlock(
            nf * 1, nf * 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv11 = self.ConvBlock(
            nf * 1, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)

    def concat_feature(self, ca, cb):
        if self.conv_type == 'partial':
            ca_feature, ca_mask = ca
            cb_feature, cb_mask = cb
            feature_cat = torch.cat((ca_feature, cb_feature), 1)
            # leave only the later mask
            return feature_cat, ca_mask
        else:
            return torch.cat((ca, cb), 1)

    def forward(self, inp):
        c8, c7, c4, c2 = inp
        # c8, c7, c4, c2, input = inp
        # img_H, img_W, img_D = input.shape[2], input.shape[3], input.shape[4]
        concat1 = self.concat_feature(c8, c4)
        d1 = self.deconv1(concat1)
        c9 = self.conv9(d1)
        # retrain for this part beacause its missing
        concat2 = self.concat_feature(c9, c2)
        d2 = self.deconv2(concat2)
        # d2 = self.deconv2(c9)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        return c11, None


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

def upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear',align_corners=True)

    return src

class SCSEModule_Linear(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                 nn.Linear(ch, ch // re, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(ch // re, ch, bias=False),
                                 nn.Sigmoid()
                                 )
        self.sSE = nn.Sequential(nn.Linear(ch, ch // re, bias=False),
                                 nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        return x * self.cSE(x) + x * self.sSE(x)


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                 nn.Conv3d(ch, ch // re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv3d(ch // re, ch, 1),
                                 nn.Sigmoid()
                                 )
        self.sSE = nn.Sequential(nn.Conv3d(ch, ch, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * (self.cSE(x).expand_as(x)) + x * (self.sSE(x).expand_as(x))
