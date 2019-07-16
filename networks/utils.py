import torch
import numpy as np
from torch.nn import Module
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.nn import Conv2d, BatchNorm2d, AvgPool2d, Linear, LeakyReLU


class Conv3x3(Module):
    """生成一个ConvBlock, 包含了conv bn relu"""
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Conv3x3, self).__init__()
        if downsample:
            stride = 2
        else:
            stride = 1

        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)

        return x

class Conv1x1(Module):
    """包含了Relu 和 bn"""
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)

        return x

class ResBlock(Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = Conv1x1(channels, int(channels/2))
        self.conv2 = Conv3x3(int(channels / 2), channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)

        return x + residual


def test_resblock():
    dummy_input = torch.randn(size=(1, 64, 256, 256))
    net = ResBlock(64)
    out = net.forward(dummy_input)
    print(out.size())
    with SummaryWriter(comment='Res') as w:
        w.add_graph(net, dummy_input)


