# -*- coding: utf-8 -*-
"""
.. codeauthor:: Matteo Sodano <matteo.sodano@igg.uni-bonn.de>
The class SqueezeAndExciteFusionAdd is copied from
https://github.com/TUI-NICR/ESANet/blob/main/src/models/rgb_depth-fusion.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class Excitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(Excitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = self.fc(x)
        y = x * weighting
        return y

class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        if rgb.sum().item() < 1e-6:
            pass
        else:
            rgb = self.se_rgb(rgb)

        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        out = rgb + depth
        return out


class ExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ExciteFusionAdd, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                           activation=activation)
        self.se_depth = Excitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        if rgb.sum().item() < 1e-6:
            pass
        else:
            rgb = self.se_rgb(rgb)

        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        out = rgb + depth
        return out


class ResidualExciteFusion(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ResidualExciteFusion, self).__init__()

        self.se_rgb = Excitation(channels_in,
                                           activation=activation)
        self.se_depth = Excitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        if rgb.sum().item() < 1e-6:
            pass
        else:
            rgb_se = self.se_rgb(rgb)

        if depth.sum().item() < 1e-6:
            pass
        else:
            depth = self.se_depth(depth)

        out = rgb + rgb_se + depth
        return out


class ViTFlattener(nn.Module):

    def __init__(self, patch_dim):
        super(ViTFlattener, self).__init__()
        self.patch_dim = patch_dim
        self.patcher = torch.nn.PixelUnshuffle(self.patch_dim)
        self.flattener = torch.nn.Flatten(-2, -1)

    def forward(self, inp):
        patches = self.patcher(inp)
        flat = self.flattener(patches)
        ViT_out = flat
        return ViT_out


class ViTUnFlattener(nn.Module):

    def __init__(self, patch_dim):
        super(ViTUnFlattener, self).__init__()
        self.patch_dim = patch_dim
        self.unpatcher = torch.nn.PixelShuffle(self.patch_dim)

    def forward(self, inp, out_shape):
        _, C, H, W = out_shape
        x = inp
        x = x.reshape(-1, C * self.patch_dim * self.patch_dim, H // self.patch_dim, W // self.patch_dim)
        x = self.unpatcher(x)
        return x


class SelfAttentionFusion(nn.Module):
    def __init__(self, patches_size, channels, bottleneck_dim=32):
        super(SelfAttentionFusion, self).__init__()

        self.patches_size = patches_size
        self.bottleneck_dim = bottleneck_dim
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim

        self.downsampler_key_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_query_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_value_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_key_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_query_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_value_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.vit_flatten = ViTFlattener(self.patches_size)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)
        self.upsampler_1 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)
        self.upsampler_2 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)


    def forward(self, rgb, depth):
        # Self-Attention for RGB
        query_rgb = self.downsampler_query_1(rgb)
        key_rgb = self.downsampler_key_1(rgb)
        value_rgb = self.downsampler_value_1(rgb)
        flattened_query_rgb = self.vit_flatten(query_rgb)
        flattened_key_rgb = self.vit_flatten(key_rgb)
        flattened_value_rgb = self.vit_flatten(value_rgb)

        QKt_rgb = torch.matmul(flattened_query_rgb, flattened_key_rgb.permute(0, 2, 1)) / self.scale
        attention_weight_rgb = self.softmax(QKt_rgb)
        output_rgb = torch.matmul(attention_weight_rgb, flattened_value_rgb)
        output_rgb = self.vit_unflatten(output_rgb, query_rgb.shape)
        output_rgb = self.upsampler_1(output_rgb)

        # Self-Attention for Depth
        query_depth = self.downsampler_query_2(depth)
        key_depth = self.downsampler_key_2(depth)
        value_depth = self.downsampler_value_2(depth)
        flattened_query_depth = self.vit_flatten(query_depth)
        flattened_key_depth = self.vit_flatten(key_depth)
        flattened_value_depth = self.vit_flatten(value_depth)

        QKt_depth = torch.matmul(flattened_query_depth, flattened_key_depth.permute(0, 2, 1)) / self.scale
        attention_weight_depth = self.softmax(QKt_depth)
        output_depth = torch.matmul(attention_weight_depth, flattened_value_depth)
        output_depth = self.vit_unflatten(output_depth, query_depth.shape)
        output_depth = self.upsampler_2(output_depth)

        # Merging
        output = output_rgb + output_depth
        return output


class ResidualAttentionFusion(nn.Module):
    def __init__(self, patches_size, channels, alpha=1., bottleneck_dim=32):
        super(ResidualAttentionFusion, self).__init__()

        self.alpha = alpha

        self.patches_size = patches_size
        self.bottleneck_dim = bottleneck_dim
        self.latent_patch_dim = self.patches_size * self.patches_size * self.bottleneck_dim

        self.downsampler_key_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_query_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_value_1 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_key_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_query_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.downsampler_value_2 = nn.Conv2d(in_channels=channels, out_channels=self.bottleneck_dim, kernel_size=1, stride=1)
        self.vit_flatten = ViTFlattener(self.patches_size)
        self.scale = torch.sqrt(torch.tensor(self.latent_patch_dim, requires_grad=False))
        self.softmax = nn.Softmax(dim=2)
        self.vit_unflatten = ViTUnFlattener(self.patches_size)
        self.upsampler_1 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)
        self.upsampler_2 = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=channels, kernel_size=1, stride=1)


    def forward(self, rgb, depth):
        # Self-Attention for RGB
        query_rgb = self.downsampler_query_1(rgb)
        key_rgb = self.downsampler_key_1(rgb)
        value_rgb = self.downsampler_value_1(rgb)
        flattened_query_rgb = self.vit_flatten(query_rgb)
        flattened_key_rgb = self.vit_flatten(key_rgb)
        flattened_value_rgb = self.vit_flatten(value_rgb)

        QKt_rgb = torch.matmul(flattened_query_rgb, flattened_key_rgb.permute(0, 2, 1)) / self.scale
        attention_weight_rgb = self.softmax(QKt_rgb)
        output_rgb = torch.matmul(attention_weight_rgb, flattened_value_rgb)
        output_rgb = self.vit_unflatten(output_rgb, query_rgb.shape)
        output_rgb = self.upsampler_1(output_rgb)

        # Self-Attention for Depth
        query_depth = self.downsampler_query_2(depth)
        key_depth = self.downsampler_key_2(depth)
        value_depth = self.downsampler_value_2(depth)
        flattened_query_depth = self.vit_flatten(query_depth)
        flattened_key_depth = self.vit_flatten(key_depth)
        flattened_value_depth = self.vit_flatten(value_depth)

        QKt_depth = torch.matmul(flattened_query_depth, flattened_key_depth.permute(0, 2, 1)) / self.scale
        attention_weight_depth = self.softmax(QKt_depth)
        output_depth = torch.matmul(attention_weight_depth, flattened_value_depth)
        output_depth = self.vit_unflatten(output_depth, query_depth.shape)
        output_depth = self.upsampler_2(output_depth)

        # Merging
        output = rgb + self.alpha * (output_rgb + output_depth)
        return output