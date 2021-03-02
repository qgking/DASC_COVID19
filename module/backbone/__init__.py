from module.backbone import resnet, vgg
from module.backbone.resnet_dilate import ResnetDilated

BACKBONE = {
    'vgg16bn': vgg.vgg16_bn,
    'vgg16': vgg.vgg16,
    'vgg19bn': vgg.vgg19_bn,
    'vgg19': vgg.vgg19,
    'resnet101': resnet.ResNet,
    'resnet50': resnet.ResNet,
    'resnet152': resnet.ResNet,
    'resnet34': resnet.ResNet,
}
BACKBONE_DILATE = {
    'resnet101': ResnetDilated,
    'resnet50': ResnetDilated,
    'resnet34': ResnetDilated,
}
