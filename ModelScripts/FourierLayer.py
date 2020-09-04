import torch
import torch.nn as nn

class FourierLayer(nn.Module):
    ''' applies a fourier transform, convolves over the transformed output,
        and then reassembles the original input from the convolved transform
        with an inverse fourier transform. '''

    def __init__(self, inChannels, outChannels, residual=False):
        super(FourierLayer,self).__init__()

        self.inChannels = inChannels
        self.outChannels = outChannels

        self.conv1 = nn.Conv3d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=outChannels, out_channels=outChannels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=outChannels, out_channels=outChannels, kernel_size=3, padding=1)

        self.activation = nn.LeakyReLU()

        self.residual = residual

    def forward(self, x):

        # fourier on input
        ft = torch.rfft(x, 2)
        # convolve over fourier transform output
        convFt1 = self.activation(self.conv1(ft))
        convFt2 = self.activation(self.conv2(convFt1))
        convFt3 = self.conv3(convFt2)

        # invert the (modified) transformed input back into it's original form
        xHat = torch.irfft(convFt3, 2)

        if self.residual == True:
            return x + xHat
        else:
            return xHat
