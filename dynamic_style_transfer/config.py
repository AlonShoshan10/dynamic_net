import argparse
import os
import utils.utils as utils


def get_configurations(parser=None):
    # set configurations here
    experiment_name = 'on_white_II_waterfall'  # write here the name of the experiment
    experiments_dir_name = os.path.join('experiments', experiment_name)
    main_style_image_name = 'on_white_II'
    tuning_blocks_style_image_name = 'waterfall'
    tuning_blocks_lower_style_image_name = 'waterfall'
    tuning_blocks_higher_style_image_name = 'waterfall'
    main_epochs = 2  # 2
    tuning_blocks_epochs = 2  # 2
    batch_size = 4
    learning_rate_main = 1e-3
    learning_rate_blocks = 1e-4
    main_content_wight = 1
    main_style_wight = 1e5

    network_version = 'normal'  # 'normal' \ 'dual'

    blocks_content_wight = 1  # set for network_version = 'normal'
    blocks_style_wight = 1e7  # set for network_version = 'normal'
    blocks_lower_content_wight = 1  # set for network_version = 'dual'
    blocks_lower_style_wight = 1e5  # set for network_version = 'dual'
    blocks_higher_content_wight = 1  # set for network_version = 'dual'
    blocks_higher_style_wight = 1e5  # set for network_version = 'dual'

    image_size = 256
    vgg_output = True
    main_style_size = None
    blocks_style_size = None
    style_wight0 = 1
    style_wight1 = 1
    style_wight2 = 1
    style_wight3 = 1
    style_wight4 = 1

    training_scheme = 'all'  # all, only_main, only_tuning_blocks, only_tuning_blocks_lower, only_tuning_blocks_higher

    checkpoint_iter = 5000
    eval_iter = 1000
    intermediate_images_iter = 500
    current_batch_eval_iter = 100

    train_data_path = 'Path to COCO2014_train'
    val_data_path = 'Path to COCO2014_val'

    model_top_params = 'main_%d_blocks_%d' % (main_style_wight, blocks_style_wight)
    checkpoint_dir = os.path.join(experiments_dir_name, 'checkpoints')
    model_save_dir = os.path.join(experiments_dir_name, 'model_dir')
    images_save_dir = os.path.join(experiments_dir_name, 'images')
    main_style_image_path = os.path.join('images', 'style_images', main_style_image_name + '.jpg')
    tuning_blocks_lower_style_image_path = os.path.join('images', 'style_images', tuning_blocks_lower_style_image_name + '.jpg')
    tuning_blocks_higher_style_image_path = os.path.join('images', 'style_images', tuning_blocks_higher_style_image_name + '.jpg')
    tuning_blocks_style_image_path = os.path.join('images', 'style_images', tuning_blocks_style_image_name + '.jpg')
    evaluation_images_path = os.path.join('images', 'evaluation_images')
    pre_trained_main_model = os.path.join(model_save_dir, 'orginal_main_latest.pth')
    pre_trained_tuning_blocks_lower = os.path.join(model_save_dir, 'tuning_blocks_lower.pth')
    pre_trained_tuning_blocks_higher = os.path.join(model_save_dir, 'tuning_blocks_higher.pth')

    # set parser
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--main_style_image_name', default=main_style_image_name)
    parser.add_argument('--main_epochs', default=main_epochs, type=int)
    parser.add_argument('--tuning_blocks_epochs', default=tuning_blocks_epochs, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--image_size', default=image_size, type=int)
    parser.add_argument('--style_size', default=main_style_size, type=int)
    parser.add_argument('--blocks_style_size', default=blocks_style_size, type=int)
    parser.add_argument('--learning_rate_main', default=learning_rate_main, type=float)
    parser.add_argument('--learning_rate_blocks', default=learning_rate_blocks, type=float)
    parser.add_argument('--main_content_wight', default=main_content_wight, type=float)
    parser.add_argument('--main_style_wight', default=main_style_wight, type=float)
    parser.add_argument('--checkpoint_iter', default=checkpoint_iter, type=int)
    parser.add_argument('--eval_iter', default=eval_iter, type=int)
    parser.add_argument('--intermediate_images_iter', default=intermediate_images_iter, type=int)
    parser.add_argument('--current_batch_eval_iter', default=current_batch_eval_iter, type=int)
    parser.add_argument('--train_data_path', default=train_data_path)
    parser.add_argument('--val_data_path', default=val_data_path)
    parser.add_argument('--model_name', default=model_top_params)
    parser.add_argument('--experiments_dir_name', default=experiments_dir_name)
    parser.add_argument('--checkpoint_dir', default=checkpoint_dir)
    parser.add_argument('--model_save_dir', default=model_save_dir)
    parser.add_argument('--images_save_dir', default=images_save_dir)
    parser.add_argument('--pre_trained_main_model', default=pre_trained_main_model)
    parser.add_argument('--main_style_image_path', default=main_style_image_path)
    parser.add_argument('--evaluation_images_path', default=evaluation_images_path)
    parser.add_argument('--vgg_output', default=vgg_output, type=lambda x:bool(utils.str2bool(x)))
    parser.add_argument('--style_wight0', default=style_wight0, type=float)
    parser.add_argument('--style_wight1', default=style_wight1, type=float)
    parser.add_argument('--style_wight2', default=style_wight2, type=float)
    parser.add_argument('--style_wight3', default=style_wight3, type=float)
    parser.add_argument('--style_wight4', default=style_wight4, type=float)
    parser.add_argument('--training_scheme', default=training_scheme)
    parser.add_argument('--network_version', default=network_version)
    if network_version is 'dual':
        parser.add_argument('--blocks_lower_content_wight', default=blocks_lower_content_wight, type=float)
        parser.add_argument('--blocks_lower_style_wight', default=blocks_lower_style_wight, type=float)
        parser.add_argument('--blocks_higher_content_wight', default=blocks_higher_content_wight, type=float)
        parser.add_argument('--blocks_higher_style_wight', default=blocks_higher_style_wight, type=float)
        parser.add_argument('--tuning_blocks_lower_style_image_name', default=tuning_blocks_lower_style_image_name)
        parser.add_argument('--tuning_blocks_higher_style_image_name', default=tuning_blocks_higher_style_image_name)
        parser.add_argument('--tuning_blocks_lower_style_image_path', default=tuning_blocks_lower_style_image_path)
        parser.add_argument('--tuning_blocks_higher_style_image_path', default=tuning_blocks_higher_style_image_path)
        parser.add_argument('--pre_trained_tuning_blocks_lower', default=pre_trained_tuning_blocks_lower)
        parser.add_argument('--pre_trained_tuning_blocks_higher', default=pre_trained_tuning_blocks_higher)
    elif network_version is 'normal':
        parser.add_argument('--blocks_content_wight', default=blocks_content_wight, type=float)
        parser.add_argument('--blocks_style_wight', default=blocks_style_wight, type=float)
        parser.add_argument('--block_style_image_name', default=tuning_blocks_style_image_name)
        parser.add_argument('--tuning_blocks_style_image_path', default=tuning_blocks_style_image_path)

    opt = parser.parse_args()
    return opt
