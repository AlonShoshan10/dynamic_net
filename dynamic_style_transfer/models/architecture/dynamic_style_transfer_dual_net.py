from torch import nn
from .main_net import MainNet
from .tuning_blocks import TuningBlockModule


class DynamicStyleTransferDual(nn.Module):
    def __init__(self):
        self.name = 'DynamicStyleTransferDual'
        super(DynamicStyleTransferDual, self).__init__()
        self.main = MainNet()
        self.tuning_blocks_lower = TuningBlockModule(b0_size=self.main.res_size, b1_size=self.main.res_size, b2_size=self.main.deconv2_output_size)
        self.tuning_blocks_higher = TuningBlockModule(b0_size=self.main.res_size, b1_size=self.main.res_size, b2_size=self.main.deconv2_output_size)
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
        if alpha_0 > 0:
            out = self.main.res3(out + alpha_0 * self.tuning_blocks_higher(out, skip='block0'))
        else:
            out = self.main.res3(out - alpha_0 * self.tuning_blocks_lower(out, skip='block0'))
        out = self.main.res4(out)
        out = self.main.res5(out)
        if alpha_1 > 0:
            out = self.relu(self.main.in4(self.main.deconv1(out + alpha_1 * self.tuning_blocks_higher(out, skip='block1'))))
        else:
            out = self.relu(self.main.in4(self.main.deconv1(out - alpha_1 * self.tuning_blocks_lower(out, skip='block1'))))
        out = self.relu(self.main.in5(self.main.deconv2(out)))
        if alpha_2 > 0:
            out = self.main.deconv3(out + alpha_2 * self.tuning_blocks_higher(out, skip='block2'))
        else:
            out = self.main.deconv3(out - alpha_2 * self.tuning_blocks_lower(out, skip='block2'))
        return out
