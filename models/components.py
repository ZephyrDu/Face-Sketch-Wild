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
    """SelfAttention
    --------------------
    replica of https://github.com/heykeetae/Self-Attention-GAN
    """

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        init_weights(self.query_conv)
        init_weights(self.key_conv)
        init_weights(self.value_conv)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
        """
        m_batch_size, c, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batch_size, -1, width * height).permute(0, 2, 1)  # B * (c//8) * (W * H)
        proj_key = self.key_conv(x).view(m_batch_size, -1, width * height)  # B * (c//8) * (W * H)
        energy = torch.bmm(proj_query, proj_key)  # B * (W * H) * (W * H)
        attention = self.softmax(energy)  # B  * N * N  (N = W*H)
        proj_value = self.value_conv(x).view(m_batch_size, -1, width * height)  # B * c * (W * H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B * c * (W * H)
        out = out.view(m_batch_size, c, width, height)  # B * c * W * H

        out = self.gamma * out + x
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
