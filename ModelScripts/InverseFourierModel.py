import torch.nn as nn
import torch
class ResNetLayer(nn.Module):

    def __init__(self, nChannels=10, dropout=0.1, batchnorm=True, activation="ReLU", initialization="xh"):
        super(ResNetLayer, self).__init__()

        self.batchnorm = batchnorm

        self.nChannels = nChannels

        self.dropout = nn.Dropout(dropout)

        self.batchnormLayer = nn.BatchNorm2d(self.nChannels)

        self.conv1 = nn.Conv2d(in_channels=self.nChannels, out_channels=self.nChannels, kernel_size=3,padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.nChannels, out_channels=self.nChannels, kernel_size=3,padding=1)

        if activation == "ReLU":
            self.activation = nn.ReLU()

        if initialization=="xh":
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        if self.batchnorm:
            return x + self.batchnormLayer(self.conv2(self.dropout(self.activation(self.conv1(self.dropout(x))))))
        else:
            return x + self.conv2(self.dropout(self.activation(self.conv1(self.dropout(x)))))

class ResNet(nn.Module):
    def __init__(self, nLayers, categories, inChannels, resChannels, dropout, batchnorm=True, activation="ReLU", initialization="xh", imageSize=32):
        super(ResNet, self).__init__()

        self.resChannels = resChannels
        self.imageSize = imageSize

        # inChannels to resChannels
        self.inToRes = nn.Conv2d(in_channels=inChannels, out_channels=resChannels, kernel_size=3, padding=1)

        # resLayers
        self.resLayers = nn.ModuleList([])
        for layer in range(nLayers):
            self.resLayers.append(ResNetLayer(resChannels, dropout, batchnorm=True, activation="ReLU", initialization="xh"))

        # outChannels to categories
        self.outToCategories = nn.Linear(in_features=(self.imageSize)*self.imageSize*resChannels, out_features=categories)


        if initialization == "xh":
            nn.init.xavier_normal_(self.inToRes.weight)
            nn.init.xavier_normal_(self.outToCategories.weight)

        elif initialization == "o":
            nn.init.orthogonal_(self.inToRes.weight)
            nn.init.orthogonal_(self.outToCategories.weight)

    def fourify(self, x):
            batchSize = x.shape[0]
            imageDimension = x.shape[-1]
            addingFactor = (self.imageSize + 1) - imageDimension
            x = torch.cat((x, torch.ones(batchSize, self.resChannels, self.imageSize, addingFactor).cuda()), dim=-1)
            x = x.reshape(batchSize, self.resChannels, self.imageSize, (self.imageSize //2) + 1, 2)
            x = torch.irfft(x, signal_ndim=2)
            return x

    def forward(self, x, fourier1=False, fourier2=False):

        batchSize = x.shape[0]
        # inChannels to resChannels
        res = self.inToRes(x)

        for layer in self.resLayers:
            if fourier1:
                if layer == len(self.resLayers) // 2:
                    res = self.fourify(res)
            res = layer(res)

        if fourier2:
            res = self.fourify(res)
#        else:
#            res = torch.cat((res, torch.ones(batchSize, self.resChannels, self.imageSize, 1).cuda()), dim=-1)
        
        out = res.view(-1, (self.imageSize) * (self.imageSize) * self.resChannels)
        categories = self.outToCategories(out)

        return categories

#model = ResNet(nLayers=20, categories=4, inChannels=3, resChannels=10, dropout=0.1, batchnorm=True, activation="ReLU", initialization="xh", imageSize=1001)
#model = model.cuda()
#print(model(torch.ones(4,3,451,451).cuda(), fourier2=True))
