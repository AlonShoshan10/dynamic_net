import torch
import torchvision
from torchvision import transforms
from models.inference_model import InferenceModel
from models.architecture.discriminator import Discriminator
from torch.optim import Adam
import os
import utils.utils as utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn


class TrainingModel(InferenceModel):
    def __init__(self, opt):
        super(TrainingModel, self).__init__(opt)
        utils.print_options(opt)
        self.train_main_loader = utils.get_data_loader(opt, train=True, main=True)
        self.train_tuning_blocks_loader = utils.get_data_loader(opt, train=True, main=False)
        self.main_disc = Discriminator().to(self.device)
        if self.opt.tuning_blocks_disc_same_as_main_disc:
            self.tuning_blocks_disc = self.main_disc
        else:
            self.tuning_blocks_disc = Discriminator().to(self.device)
        self.main_gen_optimizer = Adam(self.net.main.parameters(), lr=opt.gen_learning_rate_main, betas=(0.5, 0.999))
        self.tuning_blocks_gen_optimizer = Adam(self.net.tuning_blocks.parameters(), lr=opt.gen_learning_rate_tuning_blocks, betas=(0.5, 0.999))
        self.main_disc_optimizer = Adam(self.main_disc.parameters(), lr=opt.disc_learning_rate_main, betas=(0.5, 0.999))
        self.tuning_blocks_disc_optimizer = Adam(self.tuning_blocks_disc.parameters(), lr=opt.disc_learning_rate_tuning_blocks, betas=(0.5, 0.999))
        self.eval_tensor = torch.randn((opt.eval_noise_batch_size, self.opt.z_size)).view(-1, self.opt.z_size, 1, 1).to(self.device)
        self.criterion = nn.BCELoss().to(self.device)
        self.disc = None
        self.train_loader = None

    def train(self, main_net_training=True):
        if main_net_training:
            gen_optimizer = self.main_gen_optimizer
            disc_optimizer = self.main_disc_optimizer
            num_of_epochs = self.opt.main_epochs
            self.disc = self.main_disc
            self.train_loader = self.train_main_loader
        else:
            gen_optimizer = self.tuning_blocks_gen_optimizer
            disc_optimizer = self.tuning_blocks_disc_optimizer
            num_of_epochs = self.opt.bank_epochs
            self.disc = self.tuning_blocks_disc
            self.train_loader = self.train_tuning_blocks_loader
            for parm in self.net.main.parameters():
                parm.requires_grad = False
        self.net.train()
        self.disc.train()
        for epoch in range(num_of_epochs):
            iter_count = 0
            for batch_id, input_batch in tqdm(enumerate(self.train_loader)):
                current_batch_size = input_batch.shape[0]
                iter_count += current_batch_size
                input_batch = input_batch.to(self.device)
                # train discriminator
                disc_optimizer.zero_grad()
                disc_real_result = self.disc(input_batch).squeeze()
                label_real = torch.ones(current_batch_size).to(self.device)
                disc_real_loss = self.criterion(disc_real_result, label_real)
                label_fake = torch.zeros(current_batch_size).to(self.device)
                noise = torch.randn((current_batch_size, self.opt.z_size)).view(-1, self.opt.z_size, 1, 1).to(self.device)
                gen_result = self.forward(noise, alpha=int(not main_net_training))
                disc_fake_result = self.disc(gen_result.detach()).squeeze()
                disc_fake_loss = self.criterion(disc_fake_result, label_fake)
                disc_train_loss = disc_real_loss + disc_fake_loss
                disc_train_loss.backward()
                disc_optimizer.step()
                # train generator
                gen_optimizer.zero_grad()
                noise = torch.randn((current_batch_size, self.opt.z_size)).view(-1, self.opt.z_size, 1, 1).to(self.device)
                gen_result = self.forward(noise, alpha=int(not main_net_training))
                disc_fake_result = self.disc(gen_result).squeeze()
                gen_train_loss = self.criterion(disc_fake_result, label_real)
                gen_train_loss.backward()
                gen_optimizer.step()
                if (batch_id + 1) % self.opt.eval_iter == 0:
                    print('\n===============epoch: %d, iter: %d/%d===============' % (epoch, (batch_id + 1), len(self.train_loader)))
                    print('gen loss: %.5f, disc real loss: %.5f, disc fake loss: %.5f' % (gen_train_loss.item(), disc_real_loss.item(), disc_fake_loss.item()))
                if (batch_id + 1) % self.opt.intermediate_images_iter == 0:
                    self.show_intermediate(gen_result)
                if (batch_id + 1) % self.opt.save_image_iter == 0:
                    self.save_evaluation_images(epoch, alpha=int(not main_net_training))
            self.save_nets(epoch, main_net_training=main_net_training, latest=(epoch + 1 == num_of_epochs))

    def save_nets(self, epoch, main_net_training=True, latest=False):
        print('saving networks')
        path = self.opt.model_save_dir
        if main_net_training:
            if not latest:
                utils.save_net(self.net, path, 'gen_main_epoch_%d.pth' % epoch, self.device)
                utils.save_net(self.disc, path, 'disc_main_epoch_%d.pth' % epoch, self.device)
            else:
                utils.save_net(self.net, path, 'gen_main_latest.pth', self.device)
                utils.save_net(self.disc, path, 'disc_main_latest.pth', self.device)
                utils.save_net(self.net.main, path, 'gen_original_main_latest.pth', self.device)
        else:
            if not latest:
                utils.save_net(self.net, path, 'gen_tuning_blocks_epoch_%d.pth' % epoch, self.device)
                utils.save_net(self.disc, path, 'disc_tuning_blocks_epoch_%d.pth' % epoch, self.device)
            else:
                utils.save_net(self.net, path, 'dynamic_net.pth', self.device)
                utils.save_net(self.disc, path, 'disc_tuning_blocks_latest.pth', self.device)

    def forward(self, input_batch, alpha=0):
        return self.net(input_batch, alpha=alpha)

    def show_intermediate(self, gen_result):
        image = torchvision.utils.make_grid(self.recover_tensor(gen_result).clamp(min=0.0, max=1), nrow=16)
        image = transforms.ToPILImage()(image.cpu())
        plt.clf()
        plt.imshow(image)
        plt.pause(.001)

    def save_evaluation_images(self, epoch, alpha=0, latest=False):
        output_tensor = self.recover_tensor(self.forward(self.eval_tensor, alpha=alpha).squeeze(dim=0)).clamp(min=0.0, max=1).cpu()
        if latest:
            save_name = 'latest_alpha_%.3f.jpeg' % alpha
        else:
            save_name = 'epoch_%d_alpha_%.3f.jpeg' % (epoch, alpha)
        save_path = os.path.join(self.opt.images_save_dir, save_name)
        image = torchvision.utils.make_grid(output_tensor, nrow=16)
        utils.save_tensor_as_image(save_path, image)

    def load_pre_trained_main(self):
        if not os.path.exists(self.opt.pre_trained_original_main_model):
            print('No pre trained model')
        self.net.main.load_state_dict(torch.load(self.opt.pre_trained_original_main_model))
        print('%s pre trained main model loaded' % self.opt.pre_trained_original_main_model)
        self.net.to(self.device)
        if os.path.exists(self.opt.pre_trained_disc_model):
            print('pre trained discriminator exists')
            self.main_disc.load_state_dict(torch.load(self.opt.pre_trained_disc_model))
            print('%s pre trained discriminator loaded' % self.opt.pre_trained_disc_model)
            self.main_disc.to(self.device)