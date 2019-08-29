from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QLabel, QApplication, QPushButton, QInputDialog
from PySide2 import QtGui, QtCore
from skimage import io
from gui.base_widget import BaseWidget
import os
import utils.transformer as tr
# from models.bank_model import BankModel
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import utils.utils as utils
import config as config
from models.inference_model import InferenceModel


class ChooseStyleWidget(BaseWidget):
    def __init__(self, parent=None, main_widget=None, widget_name='choose style'):
        super(ChooseStyleWidget, self).__init__(parent=parent, widget_name=widget_name, geometry=(60,30,200,200))
        self.im_size = 128
        self.main_widget = main_widget
        pix_map_images, self.images = self.load_images()
        # dynamic style transfer
        dynamic_style_transfer_layout_txt = QHBoxLayout()
        style_transfer_txt = QLabel('Dynamic Style Transfer Networks')
        dynamic_style_transfer_layout_txt.addWidget(style_transfer_txt)
        style_transfer_txt.setAlignment(QtCore.Qt.AlignHCenter)
        dynamic_style_transfer_network_layout = QHBoxLayout()
        self.mosaic_image_label = QLabel(self)
        self.feathers_image_label = QLabel(self)
        self.udnie_image_label = QLabel(self)
        self.on_white_II_image_label = QLabel(self)
        self.autumn_landscape_image_label = QLabel(self)
        mosaic_layout, self.mosaic_button = self.make_image_and_button_layout(self.mosaic_image_label, 'Mosaic', self.on_mosaic_click, pix_map_image=pix_map_images['mosaic'])
        feathers_layout, self.feathers_button = self.make_image_and_button_layout(self.feathers_image_label, 'Feathers', self.on_feathers_click, pix_map_image=pix_map_images['feathers'])
        udnie_layout, self.udnie_button = self.make_image_and_button_layout(self.udnie_image_label, 'Udnie', self.on_udnie_click, pix_map_image=pix_map_images['udnie'])
        on_white_II_layout, self.on_white_II_button = self.make_image_and_button_layout(self.on_white_II_image_label, 'On White II', self.on_white_II_click, pix_map_image=pix_map_images['on_white_II'])
        autumn_landscape_layout, self.autumn_landscape_button = self.make_image_and_button_layout(self.autumn_landscape_image_label, 'Autumn Landscape', self.on_autumn_landscape_click, pix_map_image=pix_map_images['autumn_landscape'])
        dynamic_style_transfer_network_layout.addLayout(mosaic_layout)
        dynamic_style_transfer_network_layout.addLayout(feathers_layout)
        dynamic_style_transfer_network_layout.addLayout(udnie_layout)
        dynamic_style_transfer_network_layout.addLayout(on_white_II_layout)
        dynamic_style_transfer_network_layout.addLayout(autumn_landscape_layout)
        top_layout = QVBoxLayout()
        top_layout.addLayout(dynamic_style_transfer_layout_txt)
        top_layout.addLayout(dynamic_style_transfer_network_layout)

        # dynamic dual style transfer layout
        dynamic_dual_style_transfer_layout_txt = QHBoxLayout()
        dual_style_transfer_txt = QLabel('Dynamic Style to Style Transfer Networks')
        dynamic_dual_style_transfer_layout_txt.addWidget(dual_style_transfer_txt)
        dynamic_dual_style_transfer_layout_txt.setAlignment(QtCore.Qt.AlignHCenter)
        self.mosaic2feathers_image_label = QLabel(self)
        self.colors2mosaic_image_label = QLabel(self)
        self.udnie2feathers_image_label = QLabel(self)
        self.udnie2waterfall_image_label = QLabel(self)
        self.feathers2mosaic_image_label = QLabel(self)

        self.mosaic2rain_princess_image_label = QLabel(self)
        self.mosaic2waterfall_image_label = QLabel(self)
        self.white_II2mosaic_image_label = QLabel(self)
        self.colors2girl_image_label = QLabel(self)
        self.colors2waterfall_image_label = QLabel(self)

        self.horse2guitar_image_label = QLabel(self)
        self.mosaic2mosaic3_image_label = QLabel(self)
        self.mosaic32colors_image_label = QLabel(self)
        self.guitar2mosaic3_image_label = QLabel(self)

        mosaic2feathers_layout, self.mosaic2feathers_button = self.make_image_and_button_layout(self.mosaic2feathers_image_label, 'Mosaic -> Feathers', self.on_mosaic2feathers_click, pix_map_image=self.transformer.combine_pil_images(self.images['mosaic'], self.images['feathers'], hight=self.im_size, width=int(self.im_size/2)))
        colors2mosaic_layout, self.colors2mosaic_button = self.make_image_and_button_layout(self.colors2mosaic_image_label, 'Color Mosaic -> Mosaic', self.on_colors2mosaic_click, pix_map_image=self.transformer.combine_pil_images(self.images['colors'], self.images['mosaic'], hight=self.im_size, width=int(self.im_size/2)))
        udnie2feathers_layout, self.udnie2feathers_button = self.make_image_and_button_layout(self.udnie2feathers_image_label, 'Udnie -> Feathers', self.on_udnie2feathers_click, pix_map_image=self.transformer.combine_pil_images(self.images['udnie'], self.images['feathers'], hight=self.im_size, width=int(self.im_size/2)))
        udnie2waterfall_layout, self.udnie2waterfall_button = self.make_image_and_button_layout(self.udnie2waterfall_image_label, 'Udnie -> Waterfall', self.on_udnie2waterfall_click, pix_map_image=self.transformer.combine_pil_images(self.images['udnie'], self.images['waterfall'], hight=self.im_size, width=int(self.im_size/2)))
        feathers2mosaic_layout, self.feathers2mosaic_button = self.make_image_and_button_layout(self.feathers2mosaic_image_label, 'Feathers -> Mosaic', self.on_feathers2mosaic_click, pix_map_image=self.transformer.combine_pil_images(self.images['feathers'], self.images['mosaic'], hight=self.im_size, width=int(self.im_size/2)))

        mosaic2rain_princess_layout, self.mosaic2rain_princess_button = self.make_image_and_button_layout(self.mosaic2rain_princess_image_label, 'Mosaic -> Rain Princess', self.on_mosaic2rain_princess_click, pix_map_image=self.transformer.combine_pil_images(self.images['mosaic'], self.images['rain_princess'], hight=self.im_size, width=int(self.im_size/2)))
        mosaic2waterfall_layout, self.mosaic2waterfall_button = self.make_image_and_button_layout(self.mosaic2waterfall_image_label, 'Mosaic -> Waterfall', self.on_mosaic2waterfall_click, pix_map_image=self.transformer.combine_pil_images(self.images['mosaic'], self.images['waterfall'], hight=self.im_size, width=int(self.im_size/2)))
        white_II2mosaic_layout, self.white_II2mosaic_button = self.make_image_and_button_layout(self.white_II2mosaic_image_label, 'On White II -> Mosaic', self.on_white_II2mosaic_click, pix_map_image=self.transformer.combine_pil_images(self.images['on_white_II'], self.images['mosaic'], hight=self.im_size, width=int(self.im_size/2)))
        colors2girl_layout, self.colors2girl_button = self.make_image_and_button_layout(self.colors2girl_image_label, 'Color Mosaic -> Abstract Woman', self.on_colors2girl_click, pix_map_image=self.transformer.combine_pil_images(self.images['colors'], self.images['girl'], hight=self.im_size, width=int(self.im_size/2)))
        colors2waterfall_layout, self.colors2waterfall_button = self.make_image_and_button_layout(self.colors2waterfall_image_label, 'Color Mosaic -> Waterfall', self.on_colors2waterfall_click, pix_map_image=self.transformer.combine_pil_images(self.images['colors'], self.images['waterfall'], hight=self.im_size, width=int(self.im_size/2)))

        horse2guitar_layout, self.horse2guitar_button = self.make_image_and_button_layout(self.horse2guitar_image_label, 'Horse -> Guitar', self.on_horse2guitar_click, pix_map_image=self.transformer.combine_pil_images(self.images['horse'], self.images['guitar'], hight=self.im_size, width=int(self.im_size/2)))
        mosaic2mosaic3_layout, self.mosaic2mosaic3_button = self.make_image_and_button_layout(self.mosaic2mosaic3_image_label, 'Mosaic -> Mosaic 3', self.on_mosaic2mosaic3_click, pix_map_image=self.transformer.combine_pil_images(self.images['mosaic'], self.images['mosaic3'], hight=self.im_size, width=int(self.im_size/2)))
        mosaic32colors_layout, self.mosaic32colors_button = self.make_image_and_button_layout(self.mosaic32colors_image_label, ' Mosaic 3 -> Color Mosaic', self.on_mosaic32colors_click, pix_map_image=self.transformer.combine_pil_images(self.images['mosaic3'], self.images['colors'], hight=self.im_size, width=int(self.im_size/2)))
        guitar2mosaic3_layout, self.guitar2mosaic3_button = self.make_image_and_button_layout(self.guitar2mosaic3_image_label, 'Guitar -> Mosaic 3', self.on_guitar2mosaic3_click, pix_map_image=self.transformer.combine_pil_images(self.images['guitar'], self.images['mosaic3'], hight=self.im_size, width=int(self.im_size/2)))

        dynamic_dual_transfer_network_layout_0 = QHBoxLayout()
        dynamic_dual_transfer_network_layout_0.addLayout(udnie2feathers_layout)
        dynamic_dual_transfer_network_layout_0.addLayout(mosaic2feathers_layout)
        dynamic_dual_transfer_network_layout_0.addLayout(colors2mosaic_layout)
        dynamic_dual_transfer_network_layout_0.addLayout(udnie2waterfall_layout)
        dynamic_dual_transfer_network_layout_0.addLayout(feathers2mosaic_layout)

        dynamic_dual_transfer_network_layout_1 = QHBoxLayout()
        dynamic_dual_transfer_network_layout_1.addLayout(mosaic2rain_princess_layout)
        dynamic_dual_transfer_network_layout_1.addLayout(mosaic2waterfall_layout)
        dynamic_dual_transfer_network_layout_1.addLayout(white_II2mosaic_layout)
        dynamic_dual_transfer_network_layout_1.addLayout(colors2girl_layout)
        dynamic_dual_transfer_network_layout_1.addLayout(colors2waterfall_layout)

        dynamic_dual_transfer_network_layout_2 = QHBoxLayout()
        dynamic_dual_transfer_network_layout_2.addLayout(horse2guitar_layout)
        dynamic_dual_transfer_network_layout_2.addLayout(mosaic2mosaic3_layout)
        dynamic_dual_transfer_network_layout_2.addLayout(mosaic32colors_layout)
        dynamic_dual_transfer_network_layout_2.addLayout(guitar2mosaic3_layout)

        middle_layout = QVBoxLayout()
        middle_layout.addLayout(dynamic_dual_style_transfer_layout_txt)
        middle_layout.addLayout(dynamic_dual_transfer_network_layout_0)
        middle_layout.addLayout(dynamic_dual_transfer_network_layout_1)
        middle_layout.addLayout(dynamic_dual_transfer_network_layout_2)

        # dynamic scale style transfer layout
        dynamic_scale_style_transfer_layout_txt = QHBoxLayout()
        scale_style_transfer_txt = QLabel('Dynamic Scale Style Transfer Networks')
        dynamic_scale_style_transfer_layout_txt.addWidget(scale_style_transfer_txt)
        dynamic_scale_style_transfer_layout_txt.setAlignment(QtCore.Qt.AlignHCenter)
        dynamic_scale_transfer_network_layout = QHBoxLayout()
        self.udnie_scale_image_label = QLabel(self)
        self.waterfall_scale_image_label = QLabel(self)
        udnie_scale_layout, self.udnie_scale_button = self.make_image_and_button_layout(self.udnie_scale_image_label, 'Udnie', self.on_udnie_scale_click, pix_map_image=pix_map_images['udnie'])
        waterfall_scale_layout, self.waterfall_scale_button = self.make_image_and_button_layout(self.waterfall_scale_image_label, 'Waterfall', self.on_waterfall_scale_click, pix_map_image=pix_map_images['waterfall'])
        dynamic_scale_transfer_network_layout.addLayout(udnie_scale_layout)
        dynamic_scale_transfer_network_layout.addLayout(waterfall_scale_layout)
        bottom_layout = QVBoxLayout()
        bottom_layout.addLayout(dynamic_scale_style_transfer_layout_txt)
        bottom_layout.addLayout(dynamic_scale_transfer_network_layout)
        # set main layout
        self.layout = QVBoxLayout()
        self.layout.addLayout(top_layout)
        self.layout.addLayout(middle_layout)
        self.layout.addLayout(bottom_layout)
        self.setLayout(self.layout)
        # net init
        self.trained_nets_path = 'trained_nets'
        if not os.path.exists(self.trained_nets_path):
            self.trained_nets_path = os.path.join('..', self.trained_nets_path)
        # show widget
        self.show()

    def load_images(self):
        image_nams = ['mosaic', 'feathers', 'udnie', 'colors', 'waterfall', 'rain_princess', 'on_white_II', 'autumn_landscape', 'girl', 'horse', 'guitar', 'mosaic3']
        pix_map_images = {}
        images = {}
        for name in image_nams:
            path = os.path.join('images', 'style_images', '%s.jpg' % name)
            if not os.path.exists(path):
                path = os.path.join('..', path)
            pix_map_images[name] = self.load_pix_map(path, max_size=self.im_size)
            images[name] = Image.open(path)
        return pix_map_images, images

    def make_image_and_button_layout(self, image_ql, title, clicked_func, pix_map_image=None):
        if pix_map_image is not None:
            image_ql.setPixmap(pix_map_image)
        image_layout = self.make_image_layout(image_ql, title)
        button = QPushButton('Choose')
        button.clicked.connect(clicked_func)
        image_layout.addWidget(button)
        return image_layout, button

    def on_mosaic_click(self):
        self.main_widget.style_image = self.transformer.resize_to_max(self.images['mosaic'], 100)
        self.main_widget.set_style_image()
        print('Mosaic')
        self.choose_net('mosaic', dual=True)

    def on_feathers_click(self):
        self.main_widget.style_image = self.transformer.resize_to_max(self.images['feathers'], 100)
        self.main_widget.set_style_image()
        print('Feathers')
        self.choose_net('feathers', dual=True)

    def on_udnie_click(self):
        self.main_widget.style_image = self.transformer.resize_to_max(self.images['udnie'], 100)
        self.main_widget.set_style_image()
        print('Udnie')
        self.choose_net('udnie_1e5_5e6')

    def on_white_II_click(self):
        self.main_widget.style_image = self.transformer.resize_to_max(self.images['on_white_II'], 100)
        self.main_widget.set_style_image()
        print('On White II')
        self.choose_net('on_white_II', dual=True)

    def on_autumn_landscape_click(self):
        self.main_widget.style_image = self.transformer.resize_to_max(self.images['autumn_landscape'], 100)
        self.main_widget.set_style_image()
        print('Autumn Landscape')
        self.choose_net('autumn_landscape', dual=True)

    def on_mosaic2feathers_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['mosaic'], self.images['feathers'], pixmap=False)
        self.main_widget.set_style_image()
        print('Mosaic -> Feathers')
        self.choose_net('mosaic_1e5_to_feathers_1e6')

    def on_colors2mosaic_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['colors'], self.images['mosaic'], pixmap=False)
        self.main_widget.set_style_image()
        print('Color Mosaic -> Mosaic')
        self.choose_net('colors_1e5_None_to_mosaic_1e5')

    def on_udnie2feathers_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['udnie'], self.images['feathers'], pixmap=False)
        self.main_widget.set_style_image()
        print('Udnie -> Feathers')
        self.choose_net('udnie_256_5e5_to_feathers_1e6')

    def on_udnie2waterfall_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['udnie'], self.images['waterfall'], pixmap=False)
        self.main_widget.set_style_image()
        print('Udnie -> Waterfall')
        self.choose_net('udnie_256_5e5_to_waterfall')

    def on_feathers2mosaic_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['feathers'], self.images['mosaic'], pixmap=False)
        self.main_widget.set_style_image()
        print('Feathers -> Mosaic')
        self.choose_net('feathers_5e5_to_mosaic_1e5')

    def on_mosaic2rain_princess_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['mosaic'], self.images['rain_princess'], pixmap=False)
        self.main_widget.set_style_image()
        print('Mosaic -> Rain Princess')
        self.choose_net('mosaic_1e5_to_rain-princess_1e6')

    def on_mosaic2waterfall_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['mosaic'], self.images['waterfall'], pixmap=False)
        self.main_widget.set_style_image()
        print('Mosaic -> Waterfall')
        self.choose_net('mosaic_1e5_to_waterfall_220_5e5')

    def on_white_II2mosaic_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['on_white_II'], self.images['mosaic'], pixmap=False)
        self.main_widget.set_style_image()
        print('On White II -> Mosaic')
        self.choose_net('on_white_to_mosaic')

    def on_colors2girl_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['colors'], self.images['girl'], pixmap=False)
        self.main_widget.set_style_image()
        print('Color Mosaic -> Abstract Woman')
        self.choose_net('colors_1e5_to_girl_1e5_512')

    def on_colors2waterfall_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['colors'], self.images['waterfall'], pixmap=False)
        self.main_widget.set_style_image()
        print('Color Mosaic -> Waterfall')
        self.choose_net('colors_to_waterfall_1e6')

    def on_udnie_scale_click(self):
        self.main_widget.style_image = self.transformer.resize_to_max(self.images['udnie'], 100)
        self.main_widget.set_style_image()
        print('Udnie Scale')
        self.choose_net('udnie_5e5_None_256')

    def on_waterfall_scale_click(self):
        self.main_widget.style_image = self.transformer.resize_to_max(self.images['waterfall'], 100)
        self.main_widget.set_style_image()
        print('Waterfall Scale')
        self.choose_net('waterfall_5e5_440_220')

    ########## Update ##############
    def on_horse2guitar_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['horse'],
                                                                           self.images['guitar'], pixmap=False)
        self.main_widget.set_style_image()
        print('Horse -> Guitar')
        self.choose_net('horse_guitar')

    def on_mosaic2mosaic3_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['mosaic'],
                                                                           self.images['mosaic3'], pixmap=False)
        self.main_widget.set_style_image()
        print('Mosaic -> Mosaic 3')
        self.choose_net('mosaic_mosaic3')

    def on_mosaic32colors_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['mosaic3'],
                                                                           self.images['colors'], pixmap=False)
        self.main_widget.set_style_image()
        print('Mosaic 3 -> Colored Mosaic')
        self.choose_net('mosaic3_colors')

    def on_guitar2mosaic3_click(self):
        self.main_widget.style_image = self.transformer.combine_pil_images(self.images['guitar'],
                                                                           self.images['mosaic3'], pixmap=False)
        self.main_widget.set_style_image()
        print('Guitar -> Mosaic 3')
        self.choose_net('guitar_mosaic3')
    ################################################

    def choose_net(self, net_name, dual=False):
        self.load_net(net_name, set_net_version=('dual' if dual else 'normal'))
        if self.main_widget.dual_mode is not dual:
            self.main_widget.dual_mode = dual
            if dual:
                self.main_widget.slider_range = (-100, 100)
            else:
                self.main_widget.slider_range = (0, 100)
            self.main_widget.multi_alpha_check_box_state_changed()
        if self.main_widget.input_image is not None and self.main_widget.input_tensor is None:
            self.main_widget.calc_input_tensor()
        if self.main_widget.input_tensor is not None:
            if self.main_widget.multi_alpha:
                self.main_widget.alpha_0_slider_changed()
                self.main_widget.alpha_1_slider_changed()
                self.main_widget.alpha_2_slider_changed()
            else:
                self.main_widget.alpha_slider_changed()
            self.main_widget.run()
            self.main_widget.set_output_image()
        self.close()

    def load_net(self, net_name, set_net_version=None):
        net_path = os.path.join(self.trained_nets_path, net_name, 'model_dir', 'dynamic_net.pth')
        temp_opt = config.get_configurations()
        opt_path = os.path.join(self.trained_nets_path, net_name, 'config.txt')
        if os.path.exists(opt_path):
            opt = utils.read_config_and_arrange_opt(opt_path, temp_opt)
        else:
            opt = temp_opt
        self.main_widget.dynamic_model = InferenceModel(opt, set_net_version=set_net_version)
        self.main_widget.dynamic_model.load_network(net_path)


if __name__ == '__main__':
    # Initialise the application
    app = QApplication([])
    # Call the main widget
    ex = ChooseStyleWidget()
    app.exec_()

