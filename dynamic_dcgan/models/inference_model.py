from models.base_model import BaseModel
from models.architecture.dynamic_dcgan import DynamicDcGan
import torch


class InferenceModel(BaseModel):
    def __init__(self, opt):
        super(InferenceModel, self).__init__(opt)
        self.net = DynamicDcGan().to(self.device)

    def forward_and_recover(self, input_batch, alpha=0):
        output_batch = self.net(input_batch, alpha=alpha)
        return self.recover_tensor(output_batch)

    def load_network(self, net_path):
        self.net.load_state_dict(torch.load(net_path))

