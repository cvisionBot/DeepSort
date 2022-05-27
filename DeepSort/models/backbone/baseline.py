from ..layers.convolution import Conv2dBnAct
from ..layers.blocks import ResBlock
from ..initialize import weight_initialize

import torch
from torch import nn


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(StemBlock, self).__init__()
        self.conv = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        output = self.conv(input)
        output = self.max_pool(output)
        return output


class BaseModel(nn.Module):
    def __init__(self, in_channels, num_classes, reid_opt=False):
        super(BaseModel, self).__init__()
        self.stem = StemBlock(in_channels=in_channels, out_channels=64, kernel_size=3)
        self.avg = nn.AvgPool2d(kernel_size=(8, 4), stride=1)
        self.reid_opt = reid_opt
        # configs : in_channels, out_channels, iter, down_sample
        block1 = [64, 64, 2, 1]
        block2 = [64, 128, 2, 2]
        block3 = [128, 256, 2, 2]
        block4 = [256, 512, 2, 2]

        self.block1 = self.make_block(block1)
        self.block2 = self.make_block(block2)
        self.block3 = self.make_block(block3)
        self.block4 = self.make_block(block4)

        self.classfier = nn.Sequential(
            Conv2dBnAct(in_channels=512, out_channels=1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, num_classes, 1)
        )
    
    def forward(self, input):
        stem = self.stem(input)
        block1 = self.block1(stem)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        output = self.block4(block3)
        # output = self.avg(block4)
        if self.reid_opt:
            output = output.div(output.norm(p=2, dim=1, keepdim=True))
            return output
        output = self.classfier(output)
        return {'pred':output}
    
    def make_block(self, cfg):
        layers = []
        for i in range(cfg[2]):
            if i == 0:
                layers.append(ResBlock(cfg[0], cfg[1], stride=cfg[3]))
            else:
                layers.append(ResBlock(cfg[1], cfg[1], stride=1))
        return nn.Sequential(*layers)


def BaseNet(in_channels, classes=1000):
    model = BaseModel(in_channels=in_channels, num_classes=classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = BaseNet(in_channels=3, classes=1000)
    model(torch.rand(2, 3, 128, 64))
