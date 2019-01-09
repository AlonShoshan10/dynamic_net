from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from torchvision import datasets
import ast
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


def get_data_loader(data_set_path, batch_size, image_size, train=True, normalize=True):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    if normalize:
        transform = transforms.Compose([
            transform,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    data_set = datasets.ImageFolder(data_set_path, transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=train)
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


def calc_input_tensor(image, model, transformer):
    input_tensor = transformer.to_tensor(image).to(model.device)
    input_tensor = transformer.normalize(input_tensor)
    input_tensor = input_tensor.expand(1, -1, -1, -1)
    if input_tensor.shape[2] % 2 is not 0:
        input_tensor = input_tensor[:, :, 0:-1, :]
    if input_tensor.shape[2] % 4 is not 0:
        input_tensor = input_tensor[:, :, 1:-1, :]
    if input_tensor.shape[3] % 2 is not 0:
        input_tensor = input_tensor[:, :, :, 0:-1]
    if input_tensor.shape[3] % 4 is not 0:
        input_tensor = input_tensor[:, :, :, 1:-1]
    return input_tensor


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

