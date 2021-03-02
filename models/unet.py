from module.unet_parts_2d import *
from module.Attention_block import weights_init_normal
from module.backbone import BACKBONE, BACKBONE_DILATE
import torchvision.models as models
from functools import partial
import torch.nn.init as init
from module.Res_block import residual_block


class UNet(nn.Module):
    def __init__(self, backbone, n_channels, n_classes, downsize_nb_filters_factor=2):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.down1 = down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.down2 = down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.down3 = down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.down4 = down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.up1 = up(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.up2 = up(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.up3 = up(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.up4 = up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.outc = outconv(64 // downsize_nb_filters_factor, n_classes)
        self.apply(weights_init_normal)

    def get_backbone_layers(self):
        small_lr_layers = []
        return small_lr_layers

    def optim_parameters(self, lr):
        backbone_layer_id = [ii for m in self.get_backbone_layers() for ii in list(map(id, m.parameters()))]
        backbone_layer = filter(lambda p: id(p) in backbone_layer_id, self.parameters())
        rest_layer = filter(lambda p: id(p) not in backbone_layer_id, self.parameters())
        return [{'params': backbone_layer, 'lr': lr},
                {'params': rest_layer, 'lr': lr}]

    def forward(self, input,cam):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return [x], (x5, x5, x5), None


class UNet_dilate(nn.Module):
    def __init__(self, backbone, n_channels, n_classes, downsize_nb_filters_factor=2):
        super(UNet_dilate, self).__init__()
        downsize_nb_filters_factor = 1
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.dilate_res_1 = residual_block(input_channels=512,
                                           output_channels=512)
        self.dilate_res_1.apply(
            partial(self._nostride_dilate, dilate=2))
        self.dilate_res_2 = residual_block(input_channels=512,
                                           output_channels=512)
        self.dilate_res_2.apply(
            partial(self._nostride_dilate, dilate=4))
        downsize_nb_filters_factor = 2
        self.up1 = up(1024, 512 // downsize_nb_filters_factor)
        self.up2 = up(512, 256 // downsize_nb_filters_factor)
        self.up3 = up(256, 128 // downsize_nb_filters_factor)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.apply(self.weight_init)

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

    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def get_backbone_layers(self):
        small_lr_layers = []
        return small_lr_layers

    def forward(self, input):
        b, c, w, h = input.size()
        x = self.conv1(input)
        x = self.bn1(x)
        x1 = self.relu(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.dilate_res_1(x5)
        x7 = self.dilate_res_2(x6)
        x = self.up1(x7, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        out = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return out, x7, None
