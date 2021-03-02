import torch
import torch.nn as nn
import torch.nn.functional as F


class residual_block(nn.Module):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(self, input_channels=None, output_channels=None, kernel_size=3, stride=1, name='out'):
        super(residual_block, self).__init__()
        self.name = name
        assert isinstance(input_channels, int) and isinstance(output_channels, int)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels // 4, kernel_size, stride, padding=1)

        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, kernel_size, stride, padding=1)

        self.bn3 = nn.BatchNorm2d(output_channels // 4)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, kernel_size=1, stride=stride)

        self.conv4 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        if self.input_channels != self.output_channels or self.stride != 1:
            residual = self.conv4(x)
        # if self.name == 'out':
        #     x = torch.add([x, input])
        # else:
        #     x = add([x, input], name=name)
        out += residual
        return out
