"""
CompNet architecture (official implementation, unchanged).

CompNet = CB1 // CB2 // CB3 + FC + Dropout + ArcMargin output
Paper: https://ieeexplore.ieee.org/document/9512475

The module names match the official checkpoints exactly, so
``net.load_state_dict(torch.load(weights))`` works with ``strict=True``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class GaborConv2d(nn.Module):
    """Learnable Gabor Convolution (LGC) layer.

    Args:
        channel_in: must be 1.
        channel_out: number of learnable orientations (competitors).
        kernel_size, stride, padding: 2D convolution parameters.
        init_ratio: scale factor of the initial receptive field.
    """

    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super().__init__()

        assert channel_in == 1

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio if init_ratio > 0 else 1.0

        # Initial parameters of the Gabor function
        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        # Gaussian envelope
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False,
        )

        # Cosine carrier
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

        # Filled during forward; useful for filter visualization
        self.kernel = 0

    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()

        # [channel_out, channel_in, kernel_H, kernel_W]
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        # Rotated coordinate systems
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8 * sigma.view(-1, 1, 1, 1) ** 2)
        ) * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.genGaborBank(
            self.kernel_size, self.channel_in, self.channel_out,
            self.sigma, self.gamma, self.theta, self.f, self.psi,
        )
        self.kernel = kernel
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


class CompetitiveBlock(nn.Module):
    """Competitive Block: LGC + soft-argmax (SCC) + PPU."""

    def __init__(self, channel_in, n_competitor, ksize, stride, padding,
                 init_ratio=1, o1=32, o2=12):
        super().__init__()

        assert channel_in == 1

        self.channel_in = 1
        self.n_competitor = n_competitor
        self.init_ratio = init_ratio

        # LGC
        self.gabor_conv2d = GaborConv2d(
            channel_in=1, channel_out=n_competitor,
            kernel_size=ksize, stride=stride, padding=padding, init_ratio=init_ratio,
        )

        # Soft-argmax (SCC)
        self.a = nn.Parameter(torch.FloatTensor([1]))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.argmax = nn.Softmax(dim=1)

        # PPU
        self.conv1 = nn.Conv2d(n_competitor, o1, 5, 1, 0)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(o1, o2, 1, 1, 0)

    def forward(self, x):
        x = self.gabor_conv2d(x)
        x = (x - self.b) * self.a
        x = self.argmax(x)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        return x


class ArcMarginProduct(nn.Module):
    """Large-margin arc distance (ArcFace). Only the inference path is used here."""

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if self.training:
            assert label is not None
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine

        return output * self.s


class compnet(nn.Module):
    """CompNet: 3 competitive blocks -> FC(512) -> ArcFace."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.cb1 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=0, init_ratio=1)
        self.cb2 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=17, stride=3, padding=0, init_ratio=0.5)
        self.cb3 = CompetitiveBlock(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=0, init_ratio=0.25)

        self.fc = nn.Linear(9708, 512)
        self.drop = nn.Dropout(p=0.25)
        self.arclayer = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=False)

    def _embed(self, x):
        x1 = self.cb1(x).view(x.shape[0], -1)
        x2 = self.cb2(x).view(x.shape[0], -1)
        x3 = self.cb3(x).view(x.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)
        return self.fc(x)

    def forward(self, x, y=None):
        x = self._embed(x)
        x = self.drop(x)
        return self.arclayer(x, y)

    def getFeatureCode(self, x):
        """Return the L2-normalized 512-d feature code (official protocol)."""
        x = self._embed(x)
        return x / torch.norm(x, p=2, dim=1, keepdim=True)


# Alias used by some scripts
CompNet = compnet
