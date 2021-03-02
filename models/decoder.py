# -*- coding: utf-8 -*-
# @Time    : 19/10/14 9:50
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.unet_parts import up, outconv, double_conv
from models.ModelsGenesis.genesis import OutputTransition, UpTransition
import numpy as np
from module.aspp import SegASPPDecoder, ASPP
from module.unet_parts_2d import DecoderBlock


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("VanillaConv") != -1:
        torch.nn.init.normal_(m.featureConv.weight.data, 0.0, 0.02)
        if m.norm:
            torch.nn.init.constant_(m.norm_layer.bias.data, 0.0)
    elif classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_normal_2d(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class MedicalUNetDecoder(nn.Module):
    def __init__(self, backbone='resnet50', n_classes=3, downsize_nb_filters_factor=2):
        super(MedicalUNetDecoder, self).__init__()
        if int(backbone[6:]) > 34:
            num_layer1 = 256
            num_layer2 = 512
            num_layer3 = 1024
            num_layer4 = 2048
        else:
            num_layer1 = 64
            num_layer2 = 128
            num_layer3 = 256
            num_layer4 = 512
        self.up1 = up(num_layer4 + num_layer1, num_layer1)
        self.up2 = up(num_layer1 + 64, 64)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.outc = outconv(64, n_classes)
        self.apply(weights_init_normal)

    def forward(self, encoder_output):
        x = self.up1(encoder_output[-1], encoder_output[1])
        x = self.up2(x, encoder_output[0])
        x = self.up3(x)
        x = self.outc(x)
        return x


class GenesisUNetDecoder(nn.Module):
    def __init__(self, n_classes=3, act='relu'):
        super(GenesisUNetDecoder, self).__init__()
        self.up_tr256 = UpTransition(512, 512, 2, act)
        self.up_tr128 = UpTransition(256, 256, 1, act)
        self.up_tr64 = UpTransition(128, 128, 0, act)
        self.out_tr = OutputTransition(64, n_classes)

    def forward(self, encoder_output):
        self.out_up_256 = self.up_tr256(encoder_output.out512, encoder_output.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, encoder_output.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, encoder_output.skip_out64)
        self.out = self.out_tr(self.out_up_64)
        return self.out


class DeepLabDecoder(nn.Module):
    def __init__(self, backbone='vgg16bn', num_class=1, output_stride=16):
        super(DeepLabDecoder, self).__init__()
        batchnorm = nn.BatchNorm3d
        self.backbone = backbone
        self.aspp = ASPP(backbone, output_stride, batchnorm)
        self.decoder = SegASPPDecoder(num_class, backbone)
        self.noisy_features = False

    def set_noisy_features(self, noisy_features):
        self.noisy_features = noisy_features

    def forward(self, input):
        if self.noisy_features is True:
            noise_input = np.random.normal(loc=0.0, scale=abs(input.mean().cpu().item() * 0.05),
                                           size=input.shape).astype(np.float32)
            input = input + torch.from_numpy(noise_input).cuda()

        if 'vgg' in self.backbone:
            x, low_level_feat = input.relu5, input.relu3
        elif 'resnet' in self.backbone:
            x, low_level_feat = input.layer4, input.layer1
        else:
            raise Exception('Unknown backbone')

        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            noise_low_level_feat = np.random.normal(loc=0.0, scale=abs(low_level_feat.mean().cpu().item() *
                                                                       0.5), size=low_level_feat.shape).astype(
                np.float32)
            x += torch.from_numpy(noise_x).cuda()
            low_level_feat += torch.from_numpy(noise_low_level_feat).cuda()

        x = self.aspp(x)
        aspp = x
        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            x += torch.from_numpy(noise_x).cuda()

        low_res_x, features = self.decoder(x, low_level_feat)
        # print(low_res_x.size())
        x = F.interpolate(low_res_x, scale_factor=4, mode='trilinear', align_corners=True)
        return x, features, aspp


class DeepLab2dDecoder(nn.Module):

    def __init__(self, backbone='vgg16bn', num_class=1, output_stride=16):
        super(DeepLab2dDecoder, self).__init__()
        batchnorm = nn.BatchNorm2d
        from module.aspp2d import ASPP, SegASPPDecoder
        self.backbone = backbone
        self.aspp = ASPP(backbone, output_stride, batchnorm)
        self.decoder = SegASPPDecoder(num_class, backbone)
        self.noisy_features = False

    def set_noisy_features(self, noisy_features):
        self.noisy_features = noisy_features

    def forward(self, x, low_level_feat):
        if self.noisy_features is True:
            noise_input = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.05),
                                           size=x.shape).astype(np.float32)
            x = x + torch.from_numpy(noise_input).cuda()

        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            noise_low_level_feat = np.random.normal(loc=0.0, scale=abs(low_level_feat.mean().cpu().item() *
                                                                       0.5), size=low_level_feat.shape).astype(
                np.float32)
            x += torch.from_numpy(noise_x).cuda()
            low_level_feat += torch.from_numpy(noise_low_level_feat).cuda()

        x = self.aspp(x)
        aspp = x
        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            x += torch.from_numpy(noise_x).cuda()

        low_res_x, features = self.decoder(x, low_level_feat)
        out = F.interpolate(low_res_x, scale_factor=4, mode='bilinear', align_corners=True)
        return out, features, aspp


class DeepLabDecoder(nn.Module):

    def __init__(self, backbone='vgg16bn', num_class=1, output_stride=16):
        super(DeepLabDecoder, self).__init__()
        batchnorm = nn.BatchNorm3d
        self.backbone = backbone
        self.aspp = ASPP(backbone, output_stride, batchnorm)
        self.decoder = SegASPPDecoder(num_class, backbone)
        self.noisy_features = False

    def set_noisy_features(self, noisy_features):
        self.noisy_features = noisy_features

    def forward(self, x, low_level_feat):
        if self.noisy_features is True:
            noise_input = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.05),
                                           size=x.shape).astype(np.float32)
            x = x + torch.from_numpy(noise_input).cuda()

        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            noise_low_level_feat = np.random.normal(loc=0.0, scale=abs(low_level_feat.mean().cpu().item() *
                                                                       0.5), size=low_level_feat.shape).astype(
                np.float32)
            x += torch.from_numpy(noise_x).cuda()
            low_level_feat += torch.from_numpy(noise_low_level_feat).cuda()

        x = self.aspp(x)
        aspp = x
        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            x += torch.from_numpy(noise_x).cuda()

        low_res_x, features = self.decoder(x, low_level_feat)
        out = F.interpolate(low_res_x, scale_factor=4, mode='bilinear', align_corners=True)
        return out, features, aspp


class UnetDecoder(nn.Module):
    def __init__(
            self,
            backbone='vgg16bn',
            filter_num=32,
            num_class=1,
            attention_type='scse'
    ):
        super().__init__()
        use_batchnorm = True
        self.bb = backbone
        if backbone == 'resnet34':
            filter_list_2 = [16, 4, 2, 2, 1]
            filter_list_1 = [16 + 8, filter_list_2[0] + 4, filter_list_2[1] + 2, filter_list_2[2] + 2, 2, 1]
        if backbone == 'resnet101' or backbone == 'resnet152':
            filter_list_2 = [32, 16, 8, 2, 1]
            filter_list_1 = [32 + 64, filter_list_2[0] + 16, filter_list_2[1] + 8, filter_list_2[2] + 2, 2, 1]

        self.layer1 = DecoderBlock(filter_num * filter_list_1[0], filter_num * filter_list_2[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer2 = DecoderBlock(filter_num * filter_list_1[1], filter_num * filter_list_2[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(filter_num * filter_list_1[2], filter_num * filter_list_2[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(filter_num * filter_list_1[3], filter_num * filter_list_2[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer5 = DecoderBlock(filter_num * filter_list_1[4], filter_num * filter_list_2[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)

        self.final_conv = nn.Conv2d(filter_num * filter_list_1[5], num_class, kernel_size=(1, 1))

        self.apply(weights_init_normal_2d)

    def forward(self, backbone_features, fusion):
        if fusion is not None:
            x0, x1, x2, x3, x4 = backbone_features.layer0, backbone_features.layer1, backbone_features.layer2, \
                                 backbone_features.layer3, fusion
        else:
            x0, x1, x2, x3, x4 = backbone_features.layer0, backbone_features.layer1, backbone_features.layer2, \
                                 backbone_features.layer3, backbone_features.layer4

        x = self.layer1([x4, x3])
        x = self.layer2([x, x2])
        x = self.layer3([x, x1])
        x = self.layer4([x, x0])
        # if 'vgg' not in self.bb:
        # x = self.layer5([x, None])
        x = self.layer5([x, None])
        x = self.final_conv(x)
        return x


class UnetDecoderDilate(nn.Module):
    def __init__(
            self,
            backbone='vgg16bn',
            filter_num=32,
            num_class=1,
            attention_type='scse'
    ):
        super().__init__()
        use_batchnorm = True
        self.bb = backbone
        self.layer2 = DecoderBlock(512 + 128, 128,
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(128 + 64, 64,
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(64, 64,
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)

        self.final_conv = nn.Conv2d(64, num_class, kernel_size=(1, 1))

        self.apply(weights_init_normal)

    def forward(self, backbone_features, fusion):
        if fusion is not None:
            x0, x1, x2, x3, x4 = backbone_features.layer0, backbone_features.layer1, backbone_features.layer2, \
                                 backbone_features.layer3, fusion
        else:
            x0, x1, x2, x3, x4 = backbone_features.layer0, backbone_features.layer1, backbone_features.layer2, \
                                 backbone_features.layer3, backbone_features.layer4
        x = self.layer2([x4, x1])
        x = self.layer3([x, x0])
        x = self.layer4([x, None])
        x = self.final_conv(x)
        return x
