# -*- coding: utf-8 -*-
# @Time    : 20/4/14 14:33
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : FPN.py

# Adapted from https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Upsample(nn.Module):
    def __init__(self, scale_factor, num_channels=128):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                                 stride=1, padding=1)

    def crop_layer(self, x, target_size):
        dif = [(x.size()[2] - target_size[0]) // 2,
               (x.size()[3] - target_size[1]) // 2]
        cs = target_size
        return x[:, :, dif[0]:dif[0] + cs[0], dif[1]:dif[1] + cs[1]]

    def forward(self, x, target_size):
        out = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        out = self.crop_layer(out, target_size[2:])
        out = self.up_conv(out)
        return out


class FPN(nn.Module):
    def __init__(self, num_classes, n_channels=1, pretrained=True, freezed=False, which_resnet='resnet50'):
        super(FPN, self).__init__()
        '''FPN architecture.
               Args:
                 num_classes: (int) Number of classes present in the dataset.
                 pretrained: (bool) If True, ImageNet pretraining for ResNet is
                                    used.
                 freezed: (bool) If True, batch norm is freezed.
                 which_resnet: (str) Indicates if we use ResNet50 or ResNet101.
        '''
        self.in_planes = 64
        import torchvision.models as models
        if which_resnet == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)  # pretrained ImageNet
        elif which_resnet == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)  # pretrained ImageNet
        else:
            raise ValueError('ResNet type not recognized')
        resnet_cov1_weight = resnet.conv1.weight.data
        cov1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3,
                         bias=False)
        nv = torch.unsqueeze(torch.mean(resnet_cov1_weight, dim=1), dim=1)
        cov1.weight.data.copy_(nv)

        self.topconvs = nn.Sequential(cov1,
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1,
                                  padding=0)  # Reduce channels

        # Smooth layers
        self.smooth0 = self.lateral_smooth()
        self.smooth1 = self.lateral_smooth()
        self.smooth2 = self.lateral_smooth()
        self.smooth3 = self.lateral_smooth()

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1,
                                   padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                                   padding=0)

        # Lateral upsamples
        self.latup0 = Upsample(scale_factor=8)
        self.latup1 = Upsample(scale_factor=4)
        self.latup2 = Upsample(scale_factor=2)

        # Linear classifier
        self.classifier = nn.Conv2d(128 * 4, num_classes, kernel_size=3,
                                    stride=1, padding=1)

    def lateral_smooth(self):
        layers = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.topconvs)
        small_lr_layers.append(self.layer1)
        small_lr_layers.append(self.layer2)
        small_lr_layers.append(self.layer3)
        small_lr_layers.append(self.layer4)
        return small_lr_layers

    def optim_parameters(self, lr):
        backbone_layer_id = [ii for m in self.get_backbone_layers() for ii in list(map(id, m.parameters()))]
        backbone_layer = filter(lambda p: id(p) in backbone_layer_id, self.parameters())
        rest_layer = filter(lambda p: id(p) not in backbone_layer_id, self.parameters())
        return [{'params': backbone_layer, 'lr': lr},
                {'params': rest_layer, 'lr': 10 * lr}]

    def forward(self, x, cam):
        # Bottom-up
        c1 = self.topconvs(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5p = self.toplayer(c5)
        p4p = self._upsample_add(p5p, self.latlayer1(c4))
        p3p = self._upsample_add(p4p, self.latlayer2(c3))
        p2 = self._upsample_add(p3p, self.latlayer3(c2))
        # Lateral smooth
        p5_ = self.smooth0(p5p)
        p4_ = self.smooth1(p4p)
        p3_ = self.smooth2(p3p)
        p2_ = self.smooth3(p2)
        # Lateral upsampling
        p5 = self.latup0(p5_, p2_.size())
        p4 = self.latup1(p4_, p2_.size())
        p3 = self.latup2(p3_, p2_.size())

        out_ = [p5, p4, p3, p2_]
        out_ = torch.cat(out_, 1)
        out_ds = self.classifier(out_)
        out = F.interpolate(out_ds, scale_factor=4, mode='bilinear', align_corners=True)
        return [out], (c5, c5, c5), None


def FPN_Net(backbone='resnet50', n_channels=3, n_classes=3, pretrained=False,
            freezed=False):
    model = FPN(num_classes=n_classes,
                pretrained=pretrained, freezed=freezed, which_resnet='resnet50')
    return model
