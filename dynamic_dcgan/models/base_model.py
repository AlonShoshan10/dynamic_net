import torch
import utils.utils as utils
import os


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None

    def recover_tensor(self, image_tensor):
        mean = image_tensor.new_tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        std = image_tensor.new_tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        return (image_tensor * std) + mean

    def init_paths(self):
        utils.make_dirs(self.opt.model_save_dir)
        utils.make_dirs(self.opt.images_save_dir)

    def write_config(self):
        with open(os.path.join(self.opt.experiments_dir_name, 'config.txt'), 'w') as f:
            f.write(str(vars(self.opt)))

