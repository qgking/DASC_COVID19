# -*- coding: utf-8 -*-
# @Time    : 19/10/14 10:13
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : PretrainNet.py

from models.decoder import *
from module.attention_module import PAM_Module, CAM_Module, DANetHead
from module.modules import SELayer, SCSEModule, SCSEModule_Linear, upsample_like
from module.backbone import BACKBONE, BACKBONE_DILATE
from module.blocks import Classifier_Module
import torch
from models.u2net import *
from functools import partial
import torch.nn.init as init


# def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
#     downsample = None
#     if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
#         downsample = nn.Sequential(
#             nn.Conv2d(self.inplanes, planes * block.expansion,
#                       kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(planes * block.expansion, affine=True))
#     for i in downsample._modules['1'].parameters():
#         i.requires_grad = False
#     layers = []
#     layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
#     self.inplanes = planes * block.expansion
#     for i in range(1, blocks):
#         layers.append(block(self.inplanes, planes, dilation=dilation))
#
#     return nn.Sequential(*layers)
#
def _make_layer(inplanes, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, affine=True))
    for i in downsample._modules['1'].parameters():
        i.requires_grad = False
    layers = []
    layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, dilation=dilation))

    return nn.Sequential(*layers)


def _make_pred_layer(block, inplanes, dilation_series, padding_series, num_classes):
    return block(inplanes, dilation_series, padding_series, num_classes)


class DeepLabBase(nn.Module):
    def __init__(self, backbone, n_channels, n_classes):
        super(DeepLabBase, self).__init__()
        # self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=True)
        self.backbone = BACKBONE_DILATE[backbone](backbone=backbone, pretrained=True, dilate_scale=8)
        resnet_cov1_weight = self.backbone.topconvs[0].weight.data
        self.backbone.topconvs = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                                         bias=False),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(inplace=True),
                                               nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        nv = torch.unsqueeze(torch.mean(resnet_cov1_weight, dim=1), dim=1)
        self.backbone.topconvs[0].weight.data.copy_(nv)
        if int(backbone[6:]) > 34:
            dim = 2048
        else:
            dim = 512

        self.layer4 = _make_pred_layer(Classifier_Module, dim // 8, [6, 12, 18, 24], [6, 12, 18, 24], n_classes)
        self.layer5 = _make_pred_layer(Classifier_Module, dim // 4, [6, 12, 18, 24], [6, 12, 18, 24], n_classes)
        self.layer6 = _make_pred_layer(Classifier_Module, dim // 2, [6, 12, 18, 24], [6, 12, 18, 24], n_classes)
        self.layer7 = _make_pred_layer(Classifier_Module, dim, [6, 12, 18, 24], [6, 12, 18, 24], n_classes)
        self.layer8 = _make_pred_layer(Classifier_Module, dim, [6, 12, 18, 24], [6, 12, 18, 24], n_classes)

        self.fc8 = nn.Conv2d(dim, n_classes, 1, bias=False)
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.f8_3 = nn.Conv2d(dim // 4, 64, 1, bias=False)
        self.f8_4 = nn.Conv2d(dim // 2, 128, 1, bias=False)
        self.f9 = nn.Conv2d(192, 192, 1, bias=False)
        self.fuse = torch.nn.Conv2d(dim + 1, dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        self.seg_branch = DeepLab2dDecoder(backbone=backbone, num_class=n_classes)
        self.backbone_str = backbone

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def optim_parameters(self, lr):
        backbone_layer_id = [ii for m in self.get_backbone_layers() for ii in list(map(id, m.parameters()))]
        backbone_layer = filter(lambda p: id(p) in backbone_layer_id, self.parameters())
        rest_layer = filter(lambda p: id(p) not in backbone_layer_id, self.parameters())
        return [{'params': backbone_layer, 'lr': lr},
                {'params': rest_layer, 'lr': 10 * lr}]

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)
        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)
        return cam_rv

    def forward(self, image):
        return


class DeepLab2d_tSNE(nn.Module):
    def __init__(self, backbone, n_channels, n_classes):
        super(DeepLab2d_tSNE, self).__init__()
        self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=True)
        # self.backbone = BACKBONE_DILATE[backbone](backbone=backbone, pretrained=True, dilate_scale=8)
        resnet_cov1_weight = self.backbone.topconvs[0].weight.data
        self.backbone.topconvs = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                                         bias=False),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(inplace=True),
                                               nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        nv = torch.unsqueeze(torch.mean(resnet_cov1_weight, dim=1), dim=1)
        self.backbone.topconvs[0].weight.data.copy_(nv)

    def forward(self, image):
        backbone_out = self.backbone(image)
        features = F.adaptive_avg_pool2d(backbone_out[-1], (1, 1))
        features = features.view(features.size(0), -1)
        return features


class DeepLab2d(DeepLabBase):
    def __init__(self, backbone, n_channels, n_classes):
        super(DeepLab2d, self).__init__(backbone, n_channels, n_classes)

    def forward(self, image, cam):
        backbone_out = self.backbone(image)
        att = None
        if att is not None:
            x, low_level_feat = att, backbone_out.layer1
        else:
            x, low_level_feat = backbone_out[-1], backbone_out.layer1
        x, features, aspp = self.seg_branch(x, low_level_feat)
        return [x], (aspp, aspp, aspp), None


class DeepLab2d_CAM(nn.Module):
    def __init__(self, backbone, n_channels, n_classes):
        super(DeepLab2d_CAM, self).__init__()
        self.backbone = BACKBONE_DILATE['resnet34'](backbone='resnet34', pretrained=False, dilate_scale=8)
        self.backbone.topconvs = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                                         bias=False),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(inplace=True),
                                               nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # resnet_cov1_weight = self.backbone.topconvs[0].weight.data
        # self.backbone.topconvs = nn.Sequential(nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3,
        #                                                  bias=False),
        #                                        nn.BatchNorm2d(64),
        #                                        nn.ReLU(inplace=True),
        #                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # nv = torch.unsqueeze(torch.mean(resnet_cov1_weight, dim=1), dim=1)
        # self.backbone.topconvs[0].weight.data.copy_(nv)

        if int(backbone[6:]) > 34:
            dim = 2048
        else:
            dim = 512
        self.fc8 = nn.Conv2d(dim, n_classes, 1, bias=False)
        self.dropout7 = torch.nn.Dropout2d(0.5)

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def optim_parameters(self, lr):
        backbone_layer_id = [ii for m in self.get_backbone_layers() for ii in list(map(id, m.parameters()))]
        backbone_layer = filter(lambda p: id(p) in backbone_layer_id, self.parameters())
        rest_layer = filter(lambda p: id(p) not in backbone_layer_id, self.parameters())
        return [{'params': backbone_layer, 'lr': lr},
                {'params': rest_layer, 'lr': lr}]

    def forward_cam(self, image):
        b, c, w, h = image.size()

        backbone_out = self.backbone(image)
        cam = self.fc8(self.dropout7(backbone_out.layer4))
        return cam

    def forward(self, image):
        b, c, w, h = image.size()

        backbone_out = self.backbone(image)
        cam = self.fc8(self.dropout7(backbone_out.layer4))
        cam = F.interpolate(cam, size=(w, h), mode='bilinear', align_corners=True)
        return None, None, cam


class DeepLab2d_CAM_v19(DeepLabBase):
    def __init__(self, backbone, n_channels, n_classes):
        super(DeepLab2d_CAM_v19, self).__init__(backbone, n_channels, n_classes)

    def forward_seg(self, image, cam):
        N, C, H, W = image.size()
        backbone_out = self.backbone(image)
        # x4 = self.layer4(backbone_out.layer1)
        x7 = self.layer7(backbone_out.layer4)
        x7 = F.interpolate(x7, size=(H, W), mode='bilinear', align_corners=True)
        x8 = self.layer8(backbone_out.layer4)
        x8 = F.interpolate(x8, size=(H, W), mode='bilinear', align_corners=True)
        n, c, h, w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0

        # f8_3 = F.relu(self.f8_3(backbone_out.layer2.detach()), inplace=True)
        # f8_4 = F.relu(self.f8_4(backbone_out.layer3.detach()), inplace=True)
        # f = torch.cat([f8_3, f8_4], dim=1)
        # pcam = self.PCM(cam_d_norm, f)
        # pcam = F.interpolate(pcam, size=image.size()[2:], mode='bilinear', align_corners=True)
        pcam = F.interpolate(cam_d_norm, size=image.size()[2:], mode='bilinear', align_corners=True)
        week_cam = torch.sigmoid(pcam)
        att = None
        if att is not None:
            x, low_level_feat = att, backbone_out.layer1
        else:
            x, low_level_feat = backbone_out[-1], backbone_out.layer1
        x, features, aspp = self.seg_branch(x, low_level_feat)
        x = x * (1 + week_cam)
        return (x, x7, x8)

    def forward(self, image, cam):
        N, C, H, W = image.size()
        backbone_out = self.backbone(image)
        # x4 = self.layer4(backbone_out.layer1)
        x4 = backbone_out.layer1
        x5 = F.interpolate(backbone_out.layer2, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x6 = F.interpolate(backbone_out.layer3, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x7 = self.layer7(backbone_out.layer4)
        x7 = F.interpolate(x7, size=(H, W), mode='bilinear', align_corners=True)
        x8 = self.layer8(backbone_out.layer4)
        x8 = F.interpolate(x8, size=(H, W), mode='bilinear', align_corners=True)
        n, c, h, w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0

        # f8_3 = F.relu(self.f8_3(backbone_out.layer2.detach()), inplace=True)
        # f8_4 = F.relu(self.f8_4(backbone_out.layer3.detach()), inplace=True)
        # f = torch.cat([f8_3, f8_4], dim=1)
        # pcam = self.PCM(cam_d_norm, f)
        # pcam = F.interpolate(pcam, size=image.size()[2:], mode='bilinear', align_corners=True)
        pcam = F.interpolate(cam_d_norm, size=image.size()[2:], mode='bilinear', align_corners=True)
        week_cam = torch.sigmoid(pcam)
        att = None
        if att is not None:
            x, low_level_feat = att, backbone_out.layer1
        else:
            x, low_level_feat = backbone_out[-1], backbone_out.layer1
        x, features, aspp = self.seg_branch(x, low_level_feat)
        x = x * (1 + week_cam)
        return (x, x7, x8), (x4, x5, x6), pcam
        # return x, torch.cat([x5, x6, x7, x8], dim=1), cam
