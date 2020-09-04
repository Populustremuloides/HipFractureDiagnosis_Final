# try a normal res-net, but have a fourier layer first.

import sys
sys.path.append('../')
from ModelScripts.FourierLayer import *
from ModelScripts.ResidualLayer import *

class FourierNet(nn.Module):
    ''' creates a 'fourierNet', a residual convolutional neural network interspaced
        with fourier layers (see FourierLayer.py)
        --- the number and spacing of fourier layers is specified during initialization ---
        --- the number of residual layers is also specified, and must be larger than numFourier * fourierSpacing --- '''

    def __init__(self, numFourier, numRes, fourierSpacing, imageSize, numCategories, residual=False, numInternalChannels=10, numInputChannels=3):
        super(FourierNet, self).__init__()

        assert numFourier * fourierSpacing <= numRes, "invalid fourier layers and spacing. The following must hold: \n\n     numFourier * fourierSpacing <= numres"

        self.imageSize = imageSize

        self.numInternalChannels = numInternalChannels
        self.numInputChannels = numInputChannels

        self.numFourier = numFourier
        self.numRes = numRes
        self.fourierSpacing = fourierSpacing
        self.calculateFourierIndices() # calculate where to place the fourier layers

        # create the residual and fourier layers
        self.internalLayers = nn.ModuleList()
        for i in range(numRes):
            if i == 0:
                if i in self.fourierIndices:

                    if residual == True:
                        residualWasTrue = True
                    else:
                        residualWasTrue = False

                    if self.numInputChannels != self.numInternalChannels:
                        residual = False
                    
                    # bring the model up to internalChannels with a fourier layer (not allowing it to be residual if numInputChannels != numInternalChannels
                    self.internalLayers.append(FourierLayer(self.numInputChannels, self.numInternalChannels, residual=residual))
                    # add the residual layer behind it
                    self.internalLayers.append(ResidualLayer(self.numInternalChannels))

                    if residualWasTrue:
                        residual = True

                else:
                    # bring the model up to internalChannels with an additional layer
                    self.internalLayers.append(nn.Conv2d(in_channels=self.numInputChannels, out_channels=self.numInternalChannels, kernel_size=3, padding=1))
                    # add the residual layer
                    self.internalLayers.append(ResidualLayer(self.numInternalChannels))
            else:
                if i in self.fourierIndices:
                    self.internalLayers.append(FourierLayer(self.numInternalChannels, self.numInternalChannels, residual=residual))
                    self.internalLayers.append(ResidualLayer(self.numInternalChannels))

                else:
                    self.internalLayers.append(ResidualLayer(self.numInternalChannels))


            self.internalLayers.append(ResidualLayer(self.numInternalChannels))

        self.l1 = nn.Linear(in_features = self.imageSize * self.imageSize * self.numInternalChannels, out_features = numCategories)

    def calculateFourierIndices(self):
        self.fourierIndices = []
        currentIndex = 0
        for i in range(self.numFourier):
            self.fourierIndices.append(currentIndex)
            currentIndex += self.fourierSpacing

    def forward(self, x):
        
        for index in range(len(self.internalLayers)):
            layer = self.internalLayers[index]
            x = layer(x)

        # flatten the tensor and convert to output categories 
        x = x.view(-1, self.imageSize * self.imageSize * self.numInternalChannels)
        x = self.l1(x)
        return x
