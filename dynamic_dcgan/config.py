import argparse
import os
import utils.utils as utils


def get_configurations(parser=None):
    # set configurations here
    experiment_name = 'Female2Male'  # write here the name of the experiment
    data_set = 'celebA'
    experiments_dir_name = os.path.join('experiments', experiment_name)
    main_epochs = 20
    tuning_blocks_epochs = 20
    batch_size = 128
    z_size = 100
    gen_learning_rate_main = 0.0002
    gen_learning_rate_tuning_blocks = 0.0002
    disc_learning_rate_main = 0.0002
    disc_learning_rate_tuning_blocks = 0.0002
    image_size = 64
    tuning_blocks_disc_same_as_main_disc = False
    crop_type = '108'

    discriminator_main_attr = 'Male'
    discriminator_tuning_blocks_attr = 'Male'
    discriminator_main_attr_is = False
    discriminator_tuning_blocks_attr_is = True

    training_scheme = 'all'  # 'all', 'only_tuning_blocks', 'only_main_net

    eval_noise_batch_size = 128

    eval_iter = 100
    intermediate_images_iter = 20000
    save_image_iter = 200

    data_set_path = '/home/alon-ran/Alon/data_sets/celebA'
    attr_path = '/home/alon-ran/Alon/data_sets/celebA'

    model_save_dir = os.path.join(experiments_dir_name, 'model_dir')
    images_save_dir = os.path.join(experiments_dir_name, 'images')
    pre_trained_original_main_model = os.path.join(model_save_dir, 'original_main_latest.pth')
    pre_trained_disc_model = 'None'

    # set parser
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', default=data_set)
    parser.add_argument('--discriminator_main_attr', default=discriminator_main_attr)
    parser.add_argument('--discriminator_tuning_blocks_attr', default=discriminator_tuning_blocks_attr)
    parser.add_argument('--discriminator_main_attr_is', default=discriminator_main_attr_is, type=lambda x:bool(utils.str2bool(x)))
    parser.add_argument('--discriminator_tuning_blocks_attr_is', default=discriminator_tuning_blocks_attr_is, type=lambda x:bool(utils.str2bool(x)))
    parser.add_argument('--crop_type', default=crop_type)
    parser.add_argument('--tuning_blocks_disc_same_as_main_disc', default=tuning_blocks_disc_same_as_main_disc, type=lambda x:bool(utils.str2bool(x)))
    parser.add_argument('--main_epochs', default=main_epochs, type=int)
    parser.add_argument('--tuning_blocks_epochs', default=tuning_blocks_epochs, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--image_size', default=image_size, type=int)
    parser.add_argument('--eval_noise_batch_size', default=eval_noise_batch_size, type=int)
    parser.add_argument('--z_size', default=z_size, type=int)
    parser.add_argument('--gen_learning_rate_main', default=gen_learning_rate_main, type=float)
    parser.add_argument('--gen_learning_rate_tuning_blocks', default=gen_learning_rate_tuning_blocks, type=float)
    parser.add_argument('--disc_learning_rate_main', default=disc_learning_rate_main, type=float)
    parser.add_argument('--disc_learning_rate_tuning_blocks', default=disc_learning_rate_tuning_blocks, type=float)
    parser.add_argument('--eval_iter', default=eval_iter, type=int)
    parser.add_argument('--intermediate_images_iter', default=intermediate_images_iter, type=int)
    parser.add_argument('--save_image_iter', default=save_image_iter, type=int)
    parser.add_argument('--data_set_path', default=data_set_path)
    parser.add_argument('--attr_path', default=attr_path)
    parser.add_argument('--model_save_dir', default=model_save_dir)
    parser.add_argument('--images_save_dir', default=images_save_dir)
    parser.add_argument('--experiments_dir_name', default=experiments_dir_name)
    parser.add_argument('--pre_trained_original_main_model', default=pre_trained_original_main_model)
    parser.add_argument('--pre_trained_disc_model', default=pre_trained_disc_model)
    parser.add_argument('--training_scheme', default=training_scheme)

    opt = parser.parse_args()
    return opt
