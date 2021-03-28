from .involution_cuda import involution
from .component import *

class SketchNet(nn.Module):
    """SketchNet, transform input RGB photo to gray sketch.
    ---------------------
    A U-Net architecture similar to: https://arxiv.org/pdf/1603.08155.pdf
    Codes borrowed from: https://github.com/pytorch/examples/tree/master/fast_neural_style
    """
    def __init__(self, in_channels=3, out_channels=1, norm_type='IN'):
        super(SketchNet, self).__init__()
        # Downsample convolution layers
        self.conv1 = ConvLayer(in_channels, 32, kernel_size=3, stride=1, bias=False)
        self.norm1 = NormLayer(32, norm_type)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, bias=False)
        self.norm2 = NormLayer(64, norm_type)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2, bias=False)
        self.norm3 = NormLayer(128, norm_type)

        #
        self.invo1 = involution(in_channels,kernel_size=3, stride=1)

        # Residual layers
        self.res1 = ResidualBlock(128, norm_type)
        self.res2 = ResidualBlock(128, norm_type)
        self.res3 = ResidualBlock(128, norm_type)
        self.res4 = ResidualBlock(128, norm_type)
        self.res5 = ResidualBlock(128, norm_type)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(256, 64, kernel_size=3, stride=1, bias=False, upsample=2)
        self.norm4 = NormLayer(64, norm_type)
        self.deconv2 = UpsampleConvLayer(128, 32, kernel_size=3, stride=1, bias=False, upsample=2)
        self.norm5 = NormLayer(32, norm_type)
        self.deconv3 = ConvLayer(64, out_channels, kernel_size=3, stride=1, bias=True)

        # SA Layers
        layers = [3, 4, 6, 3]
        #[3, 4, 23, 3]
        #[3, 8, 36, 3]

        # whether to replace the 2x2 stride with a dilated convolution instead
        replace_stride_with_dilation = [False, False, False]

        self.salayer1 = self._make_layer(SABottleneck, 64, layers[0])
        self.salayer2 = self._make_layer(SABottleneck, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.salayer3 = self._make_layer(SABottleneck, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.salayer4 = self._make_layer(SABottleneck, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Non-linear layer
        self.relu = nn.ReLU(True)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, X):
        y_conv1 = self.relu(self.norm1(self.conv1(X)))
        y_conv2 = self.relu(self.norm2(self.conv2(y_conv1)))
        y_conv3 = self.relu(self.norm3(self.conv3(y_conv2)))
        y = self.res1(y_conv3)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y_deconv0 = self.res5(y)
        y_deconv0 = torch.cat((y_deconv0, y_conv3), 1)
        y_deconv1 = self.relu(self.norm4(self.deconv1(y_deconv0)))
        y_deconv1 = torch.cat((y_deconv1, y_conv2), 1)
        y_deconv2 = self.relu(self.norm5(self.deconv2(y_deconv1)))
        y_deconv2 = torch.cat((y_deconv2, y_conv1), 1)
        y = self.deconv3(y_deconv2)
        return y


class DNet(nn.Module):
    """Discriminator network.
    """
    def __init__(self, in_channels=1, norm_type='IN'):
        super(DNet, self).__init__()
        b = True if norm_type == 'none' else False
        self.net = nn.Sequential(
                ConvLayer(in_channels, 32, kernel_size=3, stride=2, bias=True),
                nn.ReLU(inplace=True),
                ConvLayer(32, 64, kernel_size=3, stride=2, bias=b),
                NormLayer(64, norm_type),
                nn.ReLU(inplace=True),
                ConvLayer(64, 128, kernel_size=3, stride=2, bias=b),
                NormLayer(128, norm_type),
                nn.ReLU(inplace=True),
                ConvLayer(128, 256, kernel_size=3, stride=2, bias=b),
                NormLayer(256, norm_type), 
                nn.ReLU(inplace=True),
                ConvLayer(256, 1, kernel_size=3, stride=1, bias=True),
                )

    def forward(self, x):
        return self.net(x) 
