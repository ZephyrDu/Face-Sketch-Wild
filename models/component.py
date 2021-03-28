import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ConvLayer(nn.Module):
    """Convolution layer with reflection padding.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class NormLayer(nn.Module):
    """Normalization layers
    -------------------
    # Args
        - channels: input channels
        - norm_type: normalization types. in: instance normalization; bn: batch normalization.
    """

    def __init__(self, channels, norm_type):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'in':
            self.norm_func = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        elif norm_type == 'bn':
            self.norm_func == nn.BatchNorm2d(channels, affine=True, track_running_stats=True)
        elif norm_type == 'none':
            self.norm_func = lambda x: x
        else:
            assert 1 == 0, 'Norm type {} not supported yet'.format(norm_type)

    def forward(self, x):
        return self.norm_func(x)


class ResidualBlock(nn.Module):
    """ResidualBlock
    ---------------------
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, norm_type='IN'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=False)
        self.norm1 = NormLayer(channels, norm_type)
        self.norm2 = NormLayer(channels, norm_type)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    --------------------
    Upsamples the input and then does a convolution.
    This method produces less checkerboard effect compared to ConvTranspose2d, according to
    http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

    """Shuffle Attention module.
    ---------------------
    Codes borrowed from: https://github.com/wofmanaf/SA-Net
    """


class SA_Layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(SA_Layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class SABottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, in_channels, channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SABottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(channels * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, channels * self.expansion)
        self.bn3 = norm_layer(channels * self.expansion)
        self.sa = SA_Layer(channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
