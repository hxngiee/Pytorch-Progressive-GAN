import os
import numpy as np

import torch
import torch.nn as nn

from layer import *

## 네트워크 구축하기
# CycleGAN
# https://arxiv.org/pdf/1703.10593.pdf


class PGGAN(nn.Module):
    def __init__(self, code_dim=512 - 10, n_label=10):
        super().__init__()

        self.label_embed = nn.Embedding(n_label, n_label)
        self.code_norm = PixelNorm()
        self.label_embed.weight.data.normal_()
        self.progression = nn.ModuleList([ConvBlock(512, 512, 4, 3, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 512, 3, 1),
                                          ConvBlock(512, 256, 3, 1),
                                          ConvBlock(256, 128, 3, 1)])

        self.to_rgb = nn.ModuleList([nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(512, 3, 1),
                                     nn.Conv2d(256, 3, 1),
                                     nn.Conv2d(128, 3, 1)])

    def forward(self, input, label, step=0, alpha=-1):
        input = self.code_norm(input)
        label = self.label_embed(label)
        out = torch.cat([input, label], 1).unsqueeze(2).unsqueeze(3)

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if i > 0 and step > 0:
                upsample = F.upsample(out, scale_factor=2)
                out = conv(upsample)

            else:
                out = conv(out)

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](upsample)
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class Discriminator(nn.Module):
    def __init__(self, n_label=10):
        super().__init__()

        self.progression = nn.ModuleList([ConvBlock(128, 256, 3, 1,
                                                    pixel_norm=False,
                                                    spectral_norm=False),
                                          ConvBlock(256, 512, 3, 1,
                                                    pixel_norm=False,
                                                    spectral_norm=False),
                                          ConvBlock(512, 512, 3, 1,
                                                    pixel_norm=False,
                                                    spectral_norm=False),
                                          ConvBlock(512, 512, 3, 1,
                                                    pixel_norm=False,
                                                    spectral_norm=False),
                                          ConvBlock(512, 512, 3, 1,
                                                    pixel_norm=False,
                                                    spectral_norm=False),
                                          ConvBlock(513, 512, 3, 1, 4, 0,
                                                    pixel_norm=False,
                                                    spectral_norm=False)])

        self.from_rgb = nn.ModuleList([nn.Conv2d(3, 128, 1),
                                       nn.Conv2d(3, 256, 1),
                                       nn.Conv2d(3, 512, 1),
                                       nn.Conv2d(3, 512, 1),
                                       nn.Conv2d(3, 512, 1),
                                       nn.Conv2d(3, 512, 1)])

        self.n_layer = len(self.progression)

        self.linear = nn.Linear(512, 1 + n_label)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                mean_std = input.std(0).mean()
                mean_std = mean_std.expand(input.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                out = F.avg_pool2d(out, 2)

                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out[:, 0], out[:, 1:]


# class CycleGAN(nn.Module):
#     def __init__(self, in_channels, out_channels, nker=64, norm='bnorm', nblk=6):
#         super(CycleGAN, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.nker = nker
#         self.norm = norm
#         self.nblk = nblk
#
#         if norm == 'bnorm':
#             self.bias = False
#         else:
#             self.bias = True
#
#         self.enc1 = CBR2d(self.in_channels, 1 * self.nker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
#         self.enc2 = CBR2d(1 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
#         self.enc3 = CBR2d(2 * self.nker, 4 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
#
#         if self.nblk:
#             res = []
#
#             for i in range(self.nblk):
#                 res += [ResBlock(4 * self.nker, 4 * self.nker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0)]
#
#             self.res = nn.Sequential(*res)
#
#         self.dec3 = DECBR2d(4 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
#         self.dec2 = DECBR2d(2 * self.nker, 1 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
#         self.dec1 = CBR2d(1 * self.nker, self.out_channels, kernel_size=7, stride=1, padding=3, norm=None, relu=None)
#
#     def forward(self, x):
#         x = self.enc1(x)
#         x = self.enc2(x)
#         x = self.enc3(x)
#
#         x = self.res(x)
#
#         x = self.dec3(x)
#         x = self.dec2(x)
#         x = self.dec1(x)
#
#         x = torch.tanh(x)
#
#         return x