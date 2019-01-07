import config
from models.training_model import TrainingModel
import utils.utils as utils

opt = config.get_configurations()

if __name__ == "__main__":
    model = TrainingModel(opt)
    model.init_paths()
    model.write_config()
    utils.print_options(opt)
    if opt.training_scheme == 'all':
        model.train(training_phase='main')
        print('Trained main network')
        if opt.network_version == 'normal':
            model.train(training_phase='tuning_blocks')
            print('Trained tuning blocks network')
        elif opt.network_version == 'dual':
            model.train(training_phase='tuning_blocks_lower')
            print('Trained tuning blocks lower network')
            model.train(training_phase='tuning_blocks_higher')
            print('Trained tuning blocks higher network')
    if opt.training_scheme == 'only_main':
        model.train(training_phase='main')
        print('Trained main network')
    model.load_pre_trained_main()
    if opt.training_scheme == 'only_tuning_blocks':
        if opt.network_version == 'normal':
            model.train(training_phase='tuning_blocks')
            print('Trained tuning blocks network')
        elif opt.network_version == 'dual':
            model.train(training_phase='tuning_blocks_lower')
            print('Trained tuning blocks lower network')
            model.train(training_phase='tuning_blocks_higher')
            print('Trained tuning blocks higher network')
    if opt.training_scheme == 'only_tuning_blocks_lower':
        model.load_pre_trained_tuning_blocks_higher()
        if opt.network_version == 'dual':
            model.train(training_phase='tuning_blocks_lower')
            print('Trained tuning blocks lower network')
    if opt.training_scheme == 'only_tuning_blocks_higher':
        model.load_pre_trained_tuning_blocks_lower()
        if opt.network_version == 'dual':
            model.train(training_phase='tuning_blocks_higher')
            print('Trained tuning blocks higher network')
