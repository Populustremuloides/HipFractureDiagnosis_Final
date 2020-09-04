# try a normal res-net, but have a fourier layer first.
import torch.nn as nn
import torch
import torchvision.datasets

class ResidualLayer(nn.Module):
    ''' residual layer for a residual neural network '''

    def __init__(self, nChannels):
        super(ResidualLayer, self).__init__()

        self.dropoutRate = 0.15
        self.dropout = nn.Dropout(self.dropoutRate)

        self.batchnormLayer = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, padding=1)

        self.activation = nn.SELU()

    def forward(self, x):
        return x + self.batchnormLayer(self.conv2(self.dropout(self.activation(self.conv1(self.dropout(x))))))
