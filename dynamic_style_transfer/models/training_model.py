import torch
import torchvision
from torchvision import transforms
from models.inference_model import InferenceModel
from torch.optim import Adam
import os
import utils.utils as utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.architecture.vgg_loss_net import LossNetwork


class TrainingModel(InferenceModel):
    def __init__(self, opt):
        super(TrainingModel, self).__init__(opt)
        self.vgg_mean = [0.485, 0.456, 0.406]
        self.vgg_std = [0.229, 0.224, 0.225]
        self.main_optimizer = Adam(self.net.main.parameters(), opt.learning_rate_main)
        if opt.network_version == 'normal':
            self.tuning_blocks_optimizer = Adam(self.net.tuning_blocks.parameters(), opt.learning_rate_blocks)
        elif opt.network_version == 'dual':
            self.tuning_blocks_lower_optimizer = Adam(self.net.tuning_blocks_lower.parameters(), opt.learning_rate_blocks)
            self.tuning_blocks_higher_optimizer = Adam(self.net.tuning_blocks_higher.parameters(), opt.learning_rate_blocks)
        self.style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.vgg_mean, std=self.vgg_std)
        ])
        self.eval_transform = transforms.ToTensor()
        if opt.vgg_output:
            self.eval_transform = transforms.Compose([
                self.eval_transform,
                transforms.Normalize(mean=self.vgg_mean, std=self.vgg_std)
            ])
        self.vgg = LossNetwork(opt).to(self.device)
        self.train_loader = utils.get_data_loader(opt.train_data_path, opt.batch_size, opt.image_size, train=True, normalize=opt.vgg_output)
        self.val_loader = utils.get_data_loader(opt.val_data_path, opt.batch_size, opt.image_size, train=False, normalize=opt.vgg_output)
        self.mse_loss = torch.nn.MSELoss().to(self.device)
        self.style_wights = [opt.style_wight0, opt.style_wight1, opt.style_wight2, opt.style_wight3]
        self.style_image_path = None

    def save_nets(self, epoch, batch_id, latest=False, training_phase='main'):
        if not latest:
            model_name = '%s_epoch_%d_iter_%d.pth' % (training_phase, epoch, batch_id)
            path = self.opt.checkpoint_dir
        else:
            if training_phase is 'main':
                model_name = '%s_net.pth' % training_phase
            else:
                model_name = 'dynamic_net.pth'
            path = self.opt.model_save_dir
        torch.save(self.net.cpu().state_dict(), os.path.join(path, model_name))
        print('saved %s to %s' % (model_name, path))
        if training_phase is 'main' and latest:
            torch.save(self.net.main.cpu().state_dict(), os.path.join(path, 'orginal_%s' % model_name))
            print('saved %s to %s' % ('orginal_%s' % model_name, path))
        if training_phase is 'tuning_blocks_lower' and latest:
            torch.save(self.net.tuning_blocks_lower.cpu().state_dict(), os.path.join(path, 'tuning_blocks_lower.pth'))
            print('saved %s to %s' % ('tuning_blocks_lower.pth', path))
        if training_phase is 'tuning_blocks_higher' and latest:
            torch.save(self.net.tuning_blocks_higher.cpu().state_dict(), os.path.join(path, 'tuning_blocks_higher.pth'))
            print('saved %s to %s' % ('tuning_blocks_higher.pth', path))
        self.net.to(self.device)

    def load_pre_trained_main(self):
        if os.path.exists(self.opt.pre_trained_main_model):
            self.net.main.load_state_dict(torch.load(self.opt.pre_trained_main_model))
            print('%s pre trained main network loaded' % self.opt.pre_trained_main_model)
            self.net.to(self.device)
        else:
            print('!!! no pre_trained_main_model !!!')
            raise Exception('no pre_trained_main_model at %s' % self.opt.pre_trained_main_model)

    def load_pre_trained_tuning_blocks_lower(self):
        if os.path.exists(self.opt.pre_trained_tuning_blocks_lower):
            self.net.tuning_blocks_lower.load_state_dict(torch.load(self.opt.pre_trained_tuning_blocks_lower))
            print('%s pre trained tuning blocks lower loaded' % self.opt.pre_trained_tuning_blocks_lower)
            self.net.to(self.device)
        else:
            print('!!! no pre_trained_tuning_blocks_lower !!!')
            raise Exception('no pre_trained_tuning_blocks_lower at %s' % self.opt.pre_trained_tuning_blocks_lower)

    def load_pre_trained_tuning_blocks_higher(self):
        if os.path.exists(self.opt.pre_trained_tuning_blocks_higher):
            self.net.tuning_blocks_higher.load_state_dict(torch.load(self.opt.pre_trained_tuning_blocks_higher))
            print('%s pre trained tuning blocks higher loaded' % self.opt.pre_trained_tuning_blocks_higher)
            self.net.to(self.device)
        else:
            print('!!! no pre_trained_tuning_blocks_higher !!!')
            raise Exception('no pre_trained_tuning_blocks_higher at %s' % self.opt.pre_trained_tuning_blocks_higher)

    def compute_gram_style(self, size):
        style = utils.load_image(self.style_image_path, size=size)
        style = self.style_transform(style)
        style = style.repeat(self.opt.batch_size, 1, 1, 1).to(self.device)
        features_style = self.vgg.get_features(style, 'style')
        gram_style = [self.gram_matrix(y) for y in features_style['style']]
        return gram_style

    def config_training_phase(self, training_phase):
        alpha_mode = None
        optimizer = None
        if training_phase == 'main':
            self.style_image_path = self.opt.main_style_image_path
            self.gram_style = self.compute_gram_style(self.opt.style_size)
            return 0, self.main_optimizer, self.opt.main_epochs, self.opt.main_content_wight, self.opt.main_style_wight
        elif training_phase == 'tuning_blocks':
            optimizer = self.tuning_blocks_optimizer
            alpha_mode = 1
            content_wight = self.opt.blocks_content_wight
            style_wight = self.opt.blocks_style_wight
            self.style_image_path = self.opt.tuning_blocks_style_image_path
        elif training_phase == 'tuning_blocks_lower':
            optimizer = self.tuning_blocks_lower_optimizer
            alpha_mode = -1
            content_wight = self.opt.blocks_lower_content_wight
            style_wight = self.opt.blocks_lower_style_wight
            self.style_image_path = self.opt.tuning_blocks_lower_style_image_path
        elif training_phase == 'tuning_blocks_higher':
            optimizer = self.tuning_blocks_higher_optimizer
            alpha_mode = 1
            content_wight = self.opt.blocks_higher_content_wight
            style_wight = self.opt.blocks_higher_style_wight
            self.style_image_path = self.opt.tuning_blocks_higher_style_image_path
        num_of_epochs = self.opt.tuning_blocks_epochs
        self.gram_style = self.compute_gram_style(self.opt.blocks_style_size)
        return alpha_mode, optimizer, num_of_epochs, content_wight, style_wight

    def train(self, training_phase='main'):
        alpha_mode, optimizer, num_of_epochs, content_wight, style_wight = self.config_training_phase(training_phase)
        for epoch in range(num_of_epochs):
            self.net.train()
            iter_count = 0
            for batch_id, (input_batch, _) in tqdm(enumerate(self.train_loader)):
                current_batch_size = input_batch.shape[0]
                iter_count += current_batch_size
                input_batch = input_batch.to(self.device)
                optimizer.zero_grad()
                output_batch = self.forward(input_batch, alpha_0=alpha_mode)
                vgg_pre_procces_output = output_batch
                vgg_pre_procces_input = input_batch
                output_features = self.vgg.get_features(vgg_pre_procces_output, 'all')
                input_features = self.vgg.get_features(vgg_pre_procces_input, 'content')
                content_loss = self.mse_loss(output_features['content'], input_features['content'])
                style_loss = 0.
                for ft_y, gm_s, s_wight in zip(output_features['style'], self.gram_style, self.style_wights):
                    gm_y = self.gram_matrix(ft_y)
                    style_loss += s_wight * self.mse_loss(gm_y, gm_s[:current_batch_size, :, :])
                total_loss = content_wight * content_loss + style_wight * style_loss
                total_loss.backward()
                optimizer.step()
                if (batch_id + 1) % self.opt.checkpoint_iter == 0:
                    self.save_nets(epoch, batch_id, training_phase=training_phase)
                if (batch_id + 1) % self.opt.intermediate_images_iter == 0:
                    self.show_intermediate(input_batch, output_batch)
                if (batch_id + 1) % self.opt.current_batch_eval_iter == 0:
                    print('epoch: %d, iter: %d/%d, content: %.7f, style: %.7f, total: %.5f' %
                          (epoch, (batch_id + 1), len(self.train_loader), content_loss.item(), style_loss.item(),
                           total_loss.item()))
                if (batch_id + 1) % self.opt.eval_iter == 0:
                    self.evaluate_iter(epoch, batch_id, alpha=alpha_mode)
                    self.save_evaluation_images(epoch, alpha=alpha_mode)
        self.save_nets(epoch, batch_id, latest=True, training_phase=training_phase)
        self.save_evaluation_images(epoch, alpha=alpha_mode, latest=True)

    def show_intermediate(self, input_batch, output_batch):
        left = torchvision.utils.make_grid(self.recover_tensor(input_batch), nrow=2)
        right = torchvision.utils.make_grid(self.recover_tensor(output_batch).clamp(min=0.0, max=1), nrow=2)
        image = torch.cat([left, right], 2)
        image = transforms.ToPILImage()(image.cpu())
        plt.clf()
        plt.imshow(image)
        plt.pause(.001)

    def evaluate_iter(self, epoch, batch_id, alpha=0):
        train_content_loss, train_style_loss = self.evaluate(self.train_loader, alpha=alpha, iters=100)
        val_content_loss, val_style_loss = self.evaluate(self.val_loader, alpha=alpha, iters=100)
        print('\n==============Epoch: %d, Iteration:%d/%d, Alpha:%.3f==============' %
              (epoch, (batch_id + 1), len(self.train_loader), alpha))
        print('Train: content: %.7f, style: %.7f, total: %.7f' %
              (train_content_loss, train_style_loss, train_content_loss + train_content_loss))
        print('Val  : content: %.7f, style: %.7f, total: %.7f' %
              (val_content_loss, val_style_loss, val_content_loss + val_content_loss))
        print('==============================================================')

    def evaluate(self, data_loader, alpha=0, iters=None):
        iter_count = 0
        batch_count = 0
        total_content_loss = 0
        total_style_loss = 0
        for batch_id, (input_batch, _) in enumerate(data_loader):
            current_batch_size = input_batch.shape[0]
            iter_count += current_batch_size
            if iters is not None and iter_count > iters:
                break
            batch_count += 1
            input_batch = input_batch.to(self.device)
            output_batch = self.forward(input_batch, alpha_0=alpha)
            vgg_pre_procces_output = output_batch
            vgg_pre_procces_input = input_batch
            output_features = self.vgg.get_features(vgg_pre_procces_output, 'all')
            input_features = self.vgg.get_features(vgg_pre_procces_input, 'content')
            content_loss = self.mse_loss(output_features['content'], input_features['content'])
            style_loss = 0.
            for ft_y, gm_s, s_wight in zip(output_features['style'], self.gram_style, self.style_wights):
                gm_y = self.gram_matrix(ft_y)
                style_loss += s_wight * self.mse_loss(gm_y, gm_s[:current_batch_size, :, :])
            total_content_loss += content_loss.item()
            total_style_loss += style_loss.item()
        return total_content_loss / batch_count, total_style_loss / batch_count

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def save_evaluation_images(self, epoch, alpha=0, latest=False):
        image_names = os.listdir(self.opt.evaluation_images_path)
        for ind, image_path in enumerate(image_names):
            eval_image = utils.load_image(os.path.join(self.opt.evaluation_images_path, image_path))
            eval_tensor = self.eval_transform(eval_image).to(self.device).expand(1, -1, -1, -1)
            output_tensor = self.recover_tensor(self.forward(eval_tensor, alpha_0=alpha).squeeze(dim=0)).clamp(min=0.0, max=1).cpu()
            if latest:
                save_name = 'latest_img_%d_alpha_%.3f.jpeg' % (ind, alpha)
            else:
                save_name = 'epoch_%d_img_%d_alpha_%.3f.jpeg' % (epoch, ind, alpha)
            save_path = os.path.join(self.opt.images_save_dir, save_name)
            utils.save_tensor_as_image(save_path, output_tensor)

    def forward(self, input_batch, alpha_0=0, alpha_1=None, alpha_2=None):
        return self.net(input_batch, alpha_0=alpha_0, alpha_1=alpha_1, alpha_2=alpha_2)

