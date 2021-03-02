# -*- coding: utf-8 -*-
# @Time    : 20/5/14 19:52
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : resnet_dilate.py
from collections import namedtuple
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

res = {
    'resnet101': models.resnet101,
    'resnet50': models.resnet50,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet152': models.resnet152,
}


class ResnetDilated(nn.Module):
    def __init__(self, backbone, pretrained=True, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial
        print('pretrained ', pretrained)
        orig_resnet = res[backbone](pretrained=pretrained)
        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.topconvs = nn.Sequential(
            OrderedDict(list(orig_resnet.named_children())[0:4]))
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.topconvs(x)
        layer0 = x
        x = self.layer1(x)
        layer1 = x
        x = self.layer2(x)
        layer2 = x
        x = self.layer3(x)
        layer3 = x
        x = self.layer4(x)
        layer4 = x
        res_outputs = namedtuple("SideOutputs", ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'])
        out = res_outputs(layer0=layer0, layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
        return out
