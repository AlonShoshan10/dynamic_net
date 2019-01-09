from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import DataLoader
import torch
import ast
from data_sets.celeb_a_data_set import CelebADataSet
import argparse

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_tensor_as_image(filename, tensor):
    img = transforms.ToPILImage()(tensor.clamp(0, 1).cpu())
    img.save(filename)


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('directory: ' + path + ' crated')
    else:
        print('directory: ' + path + ' exists')


def get_data_loader(opt, train=True, main=True):
    if not os.path.exists(opt.data_set_path):
        print(opt.data_set_path + ' doesnt exist')
        return None
    if opt.crop_type == '128':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(128),
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    elif opt.crop_type == '108':
        crop_size = 108
        re_size = 64
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(crop),
            transforms.ToPILImage(),
            transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    if opt.data_set == 'celebA':
        if main is True:
            data_set = CelebADataSet(root=opt.data_set_path, transform=transform, attr=opt.discriminator_main_attr, is_attr=opt.discriminator_main_attr_is)
        else:
            data_set = CelebADataSet(root=opt.data_set_path, transform=transform, attr=opt.discriminator_tuning_blocks_attr, is_attr=opt.discriminator_tuning_blocks_attr_is)
    data_loader = DataLoader(data_set, batch_size=opt.batch_size, shuffle=train)
    return data_loader


def read_config_and_arrange_opt(config_path, opt):
    with open(config_path) as f:
        config_dict_string = f.readline()
    config_dict = ast.literal_eval(config_dict_string)
    arrange_opt(opt, config_dict)
    return opt


def arrange_opt(opt_, config_dict):
    for key in config_dict.keys():
        opt_.__setattr__(key, config_dict[key])


def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)


def save_net(net, path, net_name, device):
    print('saving net %s to %s' % (net_name, path))
    torch.save(net.cpu().state_dict(), os.path.join(path, net_name))
    net.to(device)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

