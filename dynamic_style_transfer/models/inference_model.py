from models.base_model import BaseModel
from models.architecture.dynamic_style_transfer_net import DynamicStyleTransfer
from models.architecture.dynamic_style_transfer_dual_net import DynamicStyleTransferDual
import torch


class InferenceModel(BaseModel):
    def __init__(self, opt, set_net_version=None):
        super(InferenceModel, self).__init__(opt)
        if set_net_version is None:
            self.network_version = opt.network_version
            if self.network_version is 'dual':
                self.net = DynamicStyleTransferDual().to(self.device)
            elif self.network_version is 'normal':
                self.net = DynamicStyleTransfer().to(self.device)
        elif set_net_version == 'dual':
            self.network_version = set_net_version
            self.net = DynamicStyleTransferDual().to(self.device)
        elif set_net_version == 'normal':
            self.network_version = set_net_version
            self.net = DynamicStyleTransfer().to(self.device)


    def forward_and_recover(self, input_batch, alpha_0=0, alpha_1=None, alpha_2=None):
        output_batch = self.net(input_batch, alpha_0=alpha_0, alpha_1=alpha_1, alpha_2=alpha_2)
        return self.recover_tensor(output_batch)

    def load_network(self, net_path):
        self.net.load_state_dict(torch.load(net_path))

