from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QApplication, QPushButton
from PySide2 import QtCore
from gui.base_widget import BaseWidget
import os
from PIL import Image
import utils.utils as utils
import config as config
from models.inference_model import InferenceModel


class ChooseAttributesWidget(BaseWidget):
    def __init__(self, parent=None, main_widget=None, widget_name='choose attributes'):
        super(ChooseAttributesWidget, self).__init__(parent=parent, widget_name=widget_name, geometry=(60,30,200,200))
        self.im_size = 256
        self.main_widget = main_widget
        pix_map_images, self.images = self.load_images()
        # dynamic dcgan
        dynamic_dcgan_layout_txt = QHBoxLayout()
        style_dcgan_txt = QLabel('Dynamic DCGAN Networks')
        dynamic_dcgan_layout_txt.addWidget(style_dcgan_txt)
        style_dcgan_txt.setAlignment(QtCore.Qt.AlignHCenter)
        dynamic_dcgan_network_layout = QHBoxLayout()
        self.male2female_image_label = QLabel(self)
        self.female2male_image_label = QLabel(self)
        self.darkhair2blondhair_image_label = QLabel(self)
        self.old2young_image_label = QLabel(self)
        male2female_layout, self.male2female_button = self.make_image_and_button_layout(self.male2female_image_label, 'Male -> Female', self.on_male2female_click, pix_map_image=pix_map_images['male2female'])
        female2male_layout, self.female2male_button = self.make_image_and_button_layout(self.female2male_image_label, 'Female -> Male', self.on_female2male_click, pix_map_image=pix_map_images['female2male'])
        darkhair2blondhair_layout, self.darkhair2blondhair_button = self.make_image_and_button_layout(self.darkhair2blondhair_image_label, 'Dark Hair -> Blond Hair', self.on_darkhair2blondhair_click, pix_map_image=pix_map_images['darkhair2blondhair'])
        old2young_layout, self.on_old2young_button = self.make_image_and_button_layout(self.old2young_image_label, 'Old -> young', self.on_old2young_click, pix_map_image=pix_map_images['old2young'])
        dynamic_dcgan_network_layout.addLayout(male2female_layout)
        dynamic_dcgan_network_layout.addLayout(female2male_layout)
        dynamic_dcgan_network_layout.addLayout(darkhair2blondhair_layout)
        dynamic_dcgan_network_layout.addLayout(old2young_layout)
        top_layout = QVBoxLayout()
        top_layout.addLayout(dynamic_dcgan_layout_txt)
        top_layout.addLayout(dynamic_dcgan_network_layout)
        # set main layout
        self.layout = QVBoxLayout()
        self.layout.addLayout(top_layout)
        self.setLayout(self.layout)
        # net init
        self.trained_nets_path = 'trained_nets'
        if not os.path.exists(self.trained_nets_path):
            self.trained_nets_path = os.path.join('..', self.trained_nets_path)
        # show widget
        self.show()

    def load_images(self):
        image_nams = ['male2female', 'female2male', 'darkhair2blondhair', 'old2young']
        pix_map_images = {}
        images = {}
        for name in image_nams:
            path = os.path.join('images', 'gui_images', '%s.png' % name)
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

    def on_male2female_click(self):
        print('Male -> Female')
        self.choose_net('male2female')

    def on_female2male_click(self):
        print('Female -> Male')
        self.choose_net('female2male')

    def on_darkhair2blondhair_click(self):
        print('Dark hair -> Blond Hair')
        self.choose_net('darkhair2blondhair')

    def on_old2young_click(self):
        print('Old -> Young')
        self.choose_net('old2young')

    def choose_net(self, net_name, dual=False):
        self.load_net(net_name)
        if self.main_widget.input_tensor is not None:
            self.main_widget.run()
            self.main_widget.set_output_image()
        self.close()

    def load_net(self, net_name):
        net_path = os.path.join(self.trained_nets_path, net_name, 'model_dir', 'dynamic_net.pth')
        temp_opt = config.get_configurations()
        opt_path = os.path.join(self.trained_nets_path, net_name, 'config.txt')
        if os.path.exists(opt_path):
            opt = utils.read_config_and_arrange_opt(opt_path, temp_opt)
        else:
            opt = temp_opt
        self.main_widget.dynamic_model = InferenceModel(opt)
        self.main_widget.dynamic_model.load_network(net_path)
        self.main_widget.dynamic_model.net.train()


if __name__ == '__main__':
    # Initialise the application
    app = QApplication([])
    # Call the main widget
    ex = ChooseAttributesWidget()
    app.exec_()

