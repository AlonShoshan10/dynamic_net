from torch import nn
from .main_net import MainNet
from .tuning_blocks import TuningBlockModule


class DynamicStyleTransfer(nn.Module):
    def __init__(self):
        self.name = 'DynamicStyleTransfer'
        super(DynamicStyleTransfer, self).__init__()
        self.main = MainNet()
        self.tuning_blocks = TuningBlockModule(b0_size=self.main.res_size, b1_size=self.main.res_size, b2_size=self.main.deconv2_output_size)
        self.relu = self.main.relu

    def forward(self, x, alpha_0=0, alpha_1=None, alpha_2=None):
        if alpha_1 is None or alpha_2 is None:
            alpha_1 = alpha_0
            alpha_2 = alpha_0
        if alpha_0 == 0 and alpha_1 == 0 and alpha_2 == 0:
            return self.main(x)
        out = self.relu(self.main.in1(self.main.conv1(x)))
        out = self.relu(self.main.in2(self.main.conv2(out)))
        out = self.relu(self.main.in3(self.main.conv3(out)))
        out = self.main.res1(out)
        out = self.main.res2(out)
        out = self.main.res3(out + alpha_0 * self.tuning_blocks(out, skip='block0'))
        out = self.main.res4(out)
        out = self.main.res5(out)
        out = self.relu(self.main.in4(self.main.deconv1(out + alpha_1 * self.tuning_blocks(out, skip='block1'))))
        out = self.relu(self.main.in5(self.main.deconv2(out)))
        out = self.main.deconv3(out + alpha_2 * self.tuning_blocks(out, skip='block2'))
        return out
