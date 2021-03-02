import torch.nn as nn
from module.Res_block import residual_block
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Attention_block_simple(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block_simple, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class attention_block(nn.Module):
    """
    attention block
    https://arxiv.org/abs/1704.06904
    """

    def __init__(self, input_channels=None, output_channels=None, encoder_depth=1):
        super(attention_block, self).__init__()
        assert isinstance(input_channels, int) and isinstance(output_channels, int) and isinstance(encoder_depth, int)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.encoder_depth = encoder_depth
        self.residual_block_first_p = residual_block(input_channels, output_channels)
        self.residual_block_t = residual_block(input_channels, output_channels)
        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.residual_block_first_r = residual_block(input_channels, output_channels)
        self.residual_block_encode = residual_block(input_channels, output_channels)
        self.encode_mpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.residual_block_encode_r = residual_block(input_channels, output_channels)
        self.residual_block_decode_r = residual_block(input_channels, output_channels)
        self.residual_block_last_r = residual_block(input_channels, output_channels)
        self.out_mask_conv = nn.Sequential(
            nn.Conv3d(input_channels, input_channels, kernel_size=1, stride=1),
            nn.Conv3d(input_channels, input_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.residual_block_last_p = residual_block(input_channels, output_channels)

    def forward(self, inp):
        p = 1
        t = 2
        r = 1
        input = inp
        for i in range(p):
            input = self.residual_block_first_p(input)
        out_trunk = inp
        for j in range(t):
            out_trunk = self.residual_block_t(out_trunk)
        # Soft Mask Branch
        # encoder
        # first down sampling
        out_mask = self.mpool1(input)
        for k in range(r):
            out_mask = self.residual_block_first_r(out_mask)
        skip_connections = []
        for i in range(self.encoder_depth - 1):
            # skip connections
            output_skip_connection = self.residual_block_encode(out_mask)
            skip_connections.append(output_skip_connection)
            # down sampling
            out_mask = self.encode_mpool(out_mask)
            for _ in range(r):
                out_mask = self.residual_block_encode_r(out_mask)
            # decoder
        skip_connections = list(reversed(skip_connections))
        for i in range(self.encoder_depth - 1):
            # upsamling
            for _ in range(r):
                out_mask = self.residual_block_decode_r(out_mask)
            out_mask = F.interpolate(out_mask, scale_factor=2, mode="trilinear", align_corners=True)
            out_mask += skip_connections[i]
        # last upsampling
        for _ in range(r):
            out_mask = self.residual_block_last_r(out_mask)
        out_mask = F.interpolate(out_mask, scale_factor=2, mode="trilinear", align_corners=True)
        out_mask = self.out_mask_conv(out_mask)
        # Attention: (1 + output_soft_mask) * output_trunk
        output = (lambda x: x + 1)(out_mask)
        # output = torch.mul(output, out_trunk)
        output = output * out_trunk
        # Last Residual Block
        for i in range(p):
            output = self.residual_block_last_p(output)
        return output
