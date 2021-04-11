import torch
import torch.nn as nn


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
            self.norm_func = nn.BatchNorm2d(channels, affine=True, track_running_stats=True)
        elif norm_type == 'sn':
            self.norm_func = nn.spectral_norm()
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


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.f = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1)
        self.g = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1)
        self.h = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        init_weights(self.f)
        init_weights(self.g)
        init_weights(self.h)

    def forward(self, x):
        m_batchsize, c, width, height = x.size()
        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * C * (W * H)
        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)
        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, c, width, height)  # B * C * W * H
        out = self.gamma * self_attetion + x
        return out


"""
# Unused ChannelAttention module

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
"""
