from torch import nn
from models.architecture.main_net import MainNet
from models.architecture.tuning_blocks import TuningBlockModule


class DynamicDcGan(nn.Module):
    def __init__(self):
        super(DynamicDcGan, self).__init__()
        self.main = MainNet()
        self.tuning_blocks = TuningBlockModule(b0_size=(self.main.ngf * 2))

    def forward(self, x, alpha=0):
        if alpha == 0:
            return self.main(x)
        out = self.main.layer0(x)
        out = self.main.layer1(out)
        out = self.main.layer2(out)
        out = out + alpha * self.tuning_blocks(out, skip='block0')
        out = self.main.layer3(out)
        out = self.main.layer4(out)
        return out
