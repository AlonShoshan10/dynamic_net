import config
from models.training_model import TrainingModel

opt = config.get_configurations()

if __name__ == "__main__":
    model = TrainingModel(opt)
    model.init_paths()
    model.write_config()
    if opt.training_scheme == 'all':
        model.train(main_net_training=True)
        print('Trained main network')
        model.train(main_net_training=False)
        print('Trained bank network')
    elif opt.training_scheme == 'only_tuning_blocks':
        model.load_pre_trained_main()
        model.train(main_net_training=False)
        print('Trained bank network')
    elif opt.training_scheme == 'only_main_net':
        model.train(main_net_training=True)
        print('Trained main network')