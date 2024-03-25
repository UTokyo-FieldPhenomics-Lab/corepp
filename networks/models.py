import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
import operator

import networks.blocks as blocks

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Encoder, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class EncoderPooled(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(EncoderPooled, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class EncoderBig(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(EncoderBig, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class EncoderBigPooled(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(EncoderBigPooled, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class ERFNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        self.initial_block = blocks.DownsamplerBlock(in_channels,16)

        self.layers = [self.initial_block]

        self.layers.append(blocks.DownsamplerBlock(16,64))

        # for x in range(0, 1):    #5 times
        #    self.layers.append(blocks.non_bottleneck_1d(64, 0.03, 1)) 

        # self.layers.append(blocks.DownsamplerBlock(64,128))

        for x in range(0, 1):    #2 times
            self.layers.append(blocks.non_bottleneck_1d(64, 0.3, 2))
            self.layers.append(blocks.non_bottleneck_1d(64, 0.3, 4))
            # self.layers.append(blocks.non_bottleneck_1d(128, 0.3, 8))
            # self.layers.append(blocks.non_bottleneck_1d(128, 0.3, 16))

        self.layers.append(nn.Flatten())
        
        encoder = nn.Sequential(*self.layers)
        self.num_features = functools.reduce(operator.mul, list(encoder(torch.rand(1, *input_dim)).shape))

        self.layers.append(nn.Linear(self.num_features, self.latent_size, bias=True))
        self.encoder = nn.Sequential(*self.layers)
    
    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()


## thanks to: https://github.com/cihanongun/Point-Cloud-Autoencoder
class PointCloudEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointCloudEncoder, self).__init__()
        
        self.latent_size = out_channels

        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        return x