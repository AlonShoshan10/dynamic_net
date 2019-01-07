import torch
import utils.utils as utils
import os
import torchvision.transforms as transforms


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        if self.opt.vgg_output:
            self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def recover_tensor(self, image_tensor):
        if self.opt.vgg_output:
            mean = image_tensor.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = image_tensor.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            return (image_tensor * std) + mean
        return image_tensor

    def init_paths(self):
        utils.make_dirs(self.opt.experiments_dir_name)
        utils.make_dirs(self.opt.checkpoint_dir)
        utils.make_dirs(self.opt.model_save_dir)
        utils.make_dirs(self.opt.images_save_dir)

    def write_config(self):
        with open(os.path.join(self.opt.experiments_dir_name, 'config.txt'), 'w') as f:
            f.write(str(vars(self.opt)))

    def normalize(self, image_tensor):
        if self.opt.vgg_output:
            return self.normalize_transform(image_tensor)
        return image_tensor
