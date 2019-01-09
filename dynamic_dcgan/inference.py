import os
from models.inference_model import InferenceModel
import config
import utils.utils as utils
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse


# inference configurations #
network_name = 'female2male'
num_of_images = 6
use_saved_config = False  # use the configuration saved at training time (if saved)
alphas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.45, 0.6, 0.8, 1]
# ------------------------ #

parser = argparse.ArgumentParser()
parser.add_argument('--network_name', default=network_name)
parser.add_argument('--num_of_images', default=num_of_images, type=int)
parser.add_argument('--use_saved_config', default=use_saved_config, type=lambda x:bool(utils.str2bool(x)))
inference_opt = parser.parse_args()
network_name = inference_opt.network_name
use_saved_config = inference_opt.use_saved_config
num_of_images = inference_opt.num_of_images

networks_path = os.path.join('trained_nets', network_name)
model_path = os.path.join(networks_path, 'model_dir', 'dynamic_net.pth')
config_path = os.path.join(networks_path, 'config.txt')
save_path = os.path.join('results', 'inference_results')
if not os.path.exists(save_path):
    utils.make_dirs(save_path)

opt = config.get_configurations(parser=parser)
if use_saved_config:
    if os.path.exists(config_path):
        utils.read_config_and_arrange_opt(config_path, opt)
    else:
        raise ValueError('config_path does not exists')

dynamic_model = InferenceModel(opt)
dynamic_model.load_network(model_path)
dynamic_model.net.train()

to_tensor = transforms.ToTensor()
to_pil_image = transforms.ToPILImage()

first_image = True
input_tensor = torch.randn((128, dynamic_model.opt.z_size)).view(-1, dynamic_model.opt.z_size, 1, 1).to(dynamic_model.device)
for alpha in tqdm(alphas):
    output_tensor = dynamic_model.forward_and_recover(input_tensor.requires_grad_(False), alpha=alpha)
    image_tensor = torchvision.utils.make_grid(output_tensor[:num_of_images, :, :, :].clamp(min=0.0, max=1), nrow=1)
    if first_image:
        first_image = False
        comb_image_tensor = image_tensor
    else:
        comb_image_tensor = torch.cat([comb_image_tensor, image_tensor], dim=2)
    #output_image = to_pil_image(output_tensor[0, :, :, :].clamp(min=0.0, max=1).cpu().squeeze(dim=0))
    #output_image.save(os.path.join(save_path, '%s_%.03f.png' % (network_name, alpha)))
output_image = to_pil_image(comb_image_tensor[:, :, :].clamp(min=0.0, max=1).cpu())
output_image.save(os.path.join(save_path, '%s_grid.png' % network_name))




