import torch
import numpy as np
from networks.utils import *
from torch.nn import Module, Sequential
from torch.nn import Softmax, Linear


class Darknet53(Module):
    def __init__(self, num_classes):
        super(Darknet53, self).__init__()
        self.down1 = Sequential(
            Conv3x3(3, 32),
            Conv3x3(32, 64, True)
        )
        self.res1 = ResBlock(64)
        self.down2 = Conv3x3(64, 128, True)
        self.res2 = Sequential(
            *[ResBlock(128) for i in range(2)]
        )
        self.down3 = Conv3x3(128, 256, True)
        self.res3 = Sequential(
            *[ResBlock(256) for i in range(8)]
        )
        self.down4 = Conv3x3(256, 512, True)
        self.res4 = Sequential(
            *[ResBlock(512) for i in range(8)]
        )
        self.down5 = Conv3x3(512, 1024, True)
        self.res5 = Sequential(
            *[ResBlock(1024) for i in range(4)]
        )
        self.linear = Linear(1024, num_classes)
        self.sfmx = Softmax(dim=1)

    def forward(self, x):
        # conv part
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        x = self.down3(x)
        x = self.res3(x)
        x = self.down4(x)
        x = self.res4(x)
        x = self.down5(x)
        # linear part
        pool_size = x.size(2)
        x = F.avg_pool2d(x, kernel_size=(pool_size, ))
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = self.linear(x)
        x = self.sfmx(x)
        return x


def test_darknet53():
    dummy_input = torch.randn(size=(1, 3, 416, 416))
    net = Darknet53(1000)
    out = net.forward(dummy_input)
    print(out.size())
    with SummaryWriter(comment='Darknet53') as w:
        w.add_graph(net, dummy_input)
