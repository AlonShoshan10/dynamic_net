import torch


class MainNet(torch.nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.input_size = 3
        self.conv1_output_size = 32
        self.conv2_output_size = 64
        self.res_size = 128
        self.deconv1_output_size = 64
        self.deconv2_output_size = 32
        # Initial convolution layers
        self.conv1 = ConvLayer(self.input_size, self.conv1_output_size, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(self.conv1_output_size, affine=True)
        self.conv2 = ConvLayer(self.conv1_output_size, self.conv2_output_size, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(self.conv2_output_size, affine=True)
        self.conv3 = ConvLayer(self.conv2_output_size, self.res_size, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(self.res_size, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(self.res_size)
        self.res2 = ResidualBlock(self.res_size)
        self.res3 = ResidualBlock(self.res_size)
        self.res4 = ResidualBlock(self.res_size)
        self.res5 = ResidualBlock(self.res_size)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(self.res_size, self.deconv1_output_size, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(self.deconv1_output_size, affine=True)
        self.deconv2 = UpsampleConvLayer(self.deconv1_output_size, self.deconv2_output_size, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(self.deconv2_output_size, affine=True)
        self.deconv3 = ConvLayer(self.deconv2_output_size, self.input_size, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(
                mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out