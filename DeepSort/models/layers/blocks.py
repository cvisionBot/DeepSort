import torch
from torch import nn
from ..layers.convolution import Conv2dBnAct, Conv2dBn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2dBnAct(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride)
        self.conv2 = Conv2dBn(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.identity = Conv2dBn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        if output.size() != input.size():
            input = self.identity(input)
        output = output + input
        output = self.act(output)
        return output