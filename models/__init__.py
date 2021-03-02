from models.unet import UNet, UNet_dilate
from models.PretrainNet import *
from models.discriminator import *
from models.u2net import U2NETP, U2NET
from models.fcn import fcn8s
from models.FPN import FPN_Net
from models.advent import ADVENT
from models.AdaptSegNet import ADAPTSEGNET
from models.unetp.UNet_2Plus import UNet_2Plus
from models.unetp.UNet_3Plus import UNet_3Plus
MODELS = {
    'unet': UNet,
    'unet2plus': UNet_2Plus,
    'unet3plus': UNet_3Plus,
    'u2netp': U2NETP,
    'fpn': FPN_Net,
    'fcn8s': fcn8s,
    'deeplab_tsne': DeepLab2d_tSNE,
    'deeplabdilate2d': DeepLab2d,
    'deeplabdilate2d_cam': DeepLab2d_CAM,
    'deeplabdilate2d_camv19': DeepLab2d_CAM_v19,
    'advent': ADVENT,
    'adaptSegNet': ADAPTSEGNET,

}

DISCRIMINATOR = {
    'latent_discriminator': latent_discriminator,
    'SNPatchDiscriminator': SNTemporalPatchGANDiscriminator,
    'SDiscriminator': simple_discriminator,
    'discriminator': discriminator
}
