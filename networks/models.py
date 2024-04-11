import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import functools
import operator

import networks.blocks as blocks
from networks.resnet import ResNet18
from networks.rgb_depth_fusion import SqueezeAndExciteFusionAdd, ExciteFusionAdd, ResidualExciteFusion, SelfAttentionFusion, ResidualAttentionFusion


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
    
class DoubleEncoder(nn.Module):
    def __init__(self,
                out_channels,
                encoder_block='BasicBlock',
                bottleneck_dim=32,
                fuse_depth_in_rgb_encoder='SE-add',
                size = 256
                ):

        super(DoubleEncoder, self).__init__()
        self.latent_size = out_channels
        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder
        self.activation = nn.ReLU(inplace=True)
        self.encoder_rgb = ResNet18(
            block=encoder_block,
            pretrained_on_imagenet=False,
            activation=self.activation)
        self.encoder_depth = ResNet18(
            block=encoder_block,
            pretrained_on_imagenet=False,
            activation=self.activation,
            input_channels=1)

        h0, w0 = size // 2, size // 2
        h1, w1 = h0 // 2, w0 // 2
        h2, w2 = h1 // 2, w1 // 2
        h3, w3 = h2 // 2, w2 // 2
        h4, w4 = h3 // 2, w3 // 2
        patch_sizes = [np.gcd(h4, w4)] * 5
        
        if self.fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_2_channels_out, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_4_channels_out, activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_8_channels_out, activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_16_channels_out, activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_32_channels_out, activation=self.activation)
        elif self.fuse_depth_in_rgb_encoder == 'SelfAttention':
            self.se_layer0 = SelfAttentionFusion(patch_sizes[0], self.encoder_rgb.down_2_channels_out, bottleneck_dim)
            self.se_layer1 = SelfAttentionFusion(patch_sizes[1], self.encoder_rgb.down_4_channels_out, bottleneck_dim)
            self.se_layer2 = SelfAttentionFusion(patch_sizes[2], self.encoder_rgb.down_8_channels_out, bottleneck_dim)
            self.se_layer3 = SelfAttentionFusion(patch_sizes[3], self.encoder_rgb.down_16_channels_out, bottleneck_dim)
            self.se_layer4 = SelfAttentionFusion(patch_sizes[4], self.encoder_rgb.down_32_channels_out, bottleneck_dim)
        elif self.fuse_depth_in_rgb_encoder == 'ResidualAttention':
            self.se_layer0 = ResidualAttentionFusion(patch_sizes[0], self.encoder_rgb.down_2_channels_out, bottleneck_dim)
            self.se_layer1 = ResidualAttentionFusion(patch_sizes[1], self.encoder_rgb.down_4_channels_out, bottleneck_dim)
            self.se_layer2 = ResidualAttentionFusion(patch_sizes[2], self.encoder_rgb.down_8_channels_out, bottleneck_dim)
            self.se_layer3 = ResidualAttentionFusion(patch_sizes[3], self.encoder_rgb.down_16_channels_out, bottleneck_dim)
            self.se_layer4 = ResidualAttentionFusion(patch_sizes[4], self.encoder_rgb.down_32_channels_out, bottleneck_dim)
        elif self.fuse_depth_in_rgb_encoder == 'excite':
            self.se_layer0 = ExciteFusionAdd(self.encoder_rgb.down_2_channels_out, activation=self.activation)
            self.se_layer1 = ExciteFusionAdd(self.encoder_rgb.down_4_channels_out, activation=self.activation)
            self.se_layer2 = ExciteFusionAdd(self.encoder_rgb.down_8_channels_out, activation=self.activation)
            self.se_layer3 = ExciteFusionAdd(self.encoder_rgb.down_16_channels_out, activation=self.activation)
            self.se_layer4 = ExciteFusionAdd(self.encoder_rgb.down_32_channels_out, activation=self.activation)
        elif self.fuse_depth_in_rgb_encoder == 'ResidualExcite':
            self.se_layer0 = ResidualExciteFusion(self.encoder_rgb.down_2_channels_out, activation=self.activation)
            self.se_layer1 = ResidualExciteFusion(self.encoder_rgb.down_4_channels_out, activation=self.activation)
            self.se_layer2 = ResidualExciteFusion(self.encoder_rgb.down_8_channels_out, activation=self.activation)
            self.se_layer3 = ResidualExciteFusion(self.encoder_rgb.down_16_channels_out, activation=self.activation)
            self.se_layer4 = ResidualExciteFusion(self.encoder_rgb.down_32_channels_out, activation=self.activation)
        else:
            if self.fuse_depth_in_rgb_encoder != 'add':
                print('WARNING! You passed an invalid RGB + D fusion. Sum will be used!')
                self.fuse_depth_in_rgb_encoder == 'add'

        ## TODO this is hardcoded for the moment, I'm thinking about how to compute it based on the image size
        self.last_layer = nn.Conv2d(512,self.latent_size,1,8).cuda()

    def forward(self, rgbd):

        rgb_input = rgbd[:,:-1,:,:]
        depth_input = rgbd[:,-1,:,:].unsqueeze(1)
        rgb = self.encoder_rgb.forward_first_conv(rgb_input)
        depth = self.encoder_depth.forward_first_conv(depth_input)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer0(rgb, depth)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer1(rgb, depth)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer2(rgb, depth)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer3(rgb, depth)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer4(rgb, depth)

        out = self.last_layer(fuse)
        return out.squeeze() # TODO remove this ugly stuff
