from torch import nn


class TuningBlock(nn.Module):
    def __init__(self, input_size):
        super(TuningBlock, self).__init__()
        self.reflection_pad0 = nn.ReflectionPad2d(1)
        self.conv0 = nn.Conv2d(input_size, input_size, 3, stride=1)
        self.relu0 = nn.ReLU()
        self.reflection_pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(input_size, input_size, 3, stride=1)
        self.reflection_pad2 = nn.ReflectionPad2d(1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(input_size, input_size, 3, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv0(self.reflection_pad0(x))
        out = self.relu0(out)
        out = self.conv1(self.reflection_pad1(out))
        out = self.relu1(out)
        out = self.conv2(self.reflection_pad2(out))
        return out


class TuningBlockModule(nn.Module):
    def __init__(self, b0_size=128):
        super(TuningBlockModule, self).__init__()
        self.tuning_block0 = TuningBlock(b0_size)

    def forward(self, x, skip='block0'):
        if skip == 'block0':
            return self.tuning_block0(x)
