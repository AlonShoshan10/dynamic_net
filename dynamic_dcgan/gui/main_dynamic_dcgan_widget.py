from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QCheckBox, QLabel, QApplication, QPushButton, QLayout
from PySide2 import QtCore
from gui.base_widget import BaseWidget
import os
import torch
from gui.choose_attributes_widget import ChooseAttributesWidget
import torchvision
from utils.transformer import Transformer


class MainDynamicDcGanWidget(BaseWidget):
    def __init__(self, parent=None, app_name='Dynamic DCGAN', manual_seed=None):
        super(MainDynamicDcGanWidget, self).__init__(parent=parent, widget_name=app_name)
        self.choose_network()
        # make main layout
        self.main_layout = QHBoxLayout()
        self.generated_image_label = QLabel(self)
        generated_image_layout = self.make_image_layout(self.generated_image_label, 'Generated Images')
        self.generated_image_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.main_layout.addLayout(generated_image_layout)
        # make sliders
        self.multi_alpha = False
        self.alpha_0 = 0
        self.slider_layout = QVBoxLayout()
        self.make_sliders()
        # make check boxes
        check_boxes_layout = QHBoxLayout()
        self.show_8_check_box = QCheckBox("Show 8 images")
        self.show_16_check_box = QCheckBox("Show 16 images")
        self.show_24_check_box = QCheckBox("Show 24 images")
        self.show_8_check_box.stateChanged.connect(self.show_8_check_box_state_changed)
        self.show_16_check_box.stateChanged.connect(self.show_16_check_box_state_changed)
        self.show_24_check_box.stateChanged.connect(self.show_24_check_box_state_changed)
        check_boxes_layout.addWidget(self.show_8_check_box)
        check_boxes_layout.addWidget(self.show_16_check_box)
        check_boxes_layout.addWidget(self.show_24_check_box)
        # make generation Button
        generation_button = QPushButton('Generate')
        generation_button.clicked.connect(self.on_generation_button_click)
        # make save image Button
        save_image_button = QPushButton('Save Image')
        save_image_button.clicked.connect(self.on_save_image_click)
        # make change attributes Button
        change_attributes_button = QPushButton('Change Attributes')
        change_attributes_button.clicked.connect(self.on_change_attributes_click)
        # make buttons layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(generation_button)
        self.buttons_layout.addWidget(change_attributes_button)
        self.buttons_layout.addWidget(save_image_button)
        # widget layout
        self.layout = QVBoxLayout()
        self.layout.addLayout(self.main_layout)
        self.layout.addLayout(self.slider_layout)
        self.layout.addLayout(check_boxes_layout)
        self.layout.addLayout(self.buttons_layout)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(self.layout)
        # net init
        self.dynamic_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_tensor = None
        self.output_tensor = None
        self.output_image = None
        self.saved_images_count = 0
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
        self.transformer = Transformer()
        # show widget
        self.show()

    def make_input_and_style_layout(self):
        self.input_image_label = QLabel(self)
        input_image_layout = self.make_image_layout(self.input_image_label, 'Input Image')
        self.style_image_label = QLabel(self)
        style_image_layout = self.make_image_layout(self.style_image_label, 'style Image')
        self.input_and_style_layout.addLayout(input_image_layout)
        self.input_and_style_layout.addLayout(style_image_layout)

    # sliders functions
    def make_sliders(self):
        alpha_slider_layout, self.alpha_slider, self.alpha_txt = self.make_slider_layout(self.alpha_slider_changed, val=self.alpha_0)
        self.alpha_txt.setText(U"\U0001D6C2 = %.3f" % self.alpha_0)
        self.slider_layout.addLayout(alpha_slider_layout)

    def alpha_slider_changed(self):
        self.alpha_0 = (self.alpha_slider.value() / 100) ** 2
        if self.alpha_0 > 0.96:
            self.alpha_0 = 1
        self.alpha_txt.setText(U"\U0001D6C2 0 = %.3f" % self.alpha_0)
        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    # images functions
    def generate(self):
        return torch.randn((128, self.dynamic_model.opt.z_size)).view(-1, self.dynamic_model.opt.z_size, 1, 1).to(self.device)

    def set_output_image(self):
        nrow = 8
        num_of_images = 1
        if self.show_8_check_box.isChecked():
            num_of_images = 8
        elif self.show_16_check_box.isChecked():
            num_of_images = 16
        elif self.show_24_check_box.isChecked():
            num_of_images = 24
        image = torchvision.utils.make_grid(self.output_tensor[:num_of_images, :, :, :].clamp(min=0.0, max=1), nrow=nrow)
        self.output_image = self.transformer.to_pil_image(image.cpu())
        # self.output_image = self.output_image.resize((int(self.output_image.width * 1.5), int(self.output_image.height * 1.5)), Image.ANTIALIAS)
        pix_map = self.transformer.pil2pixmap(self.output_image)
        self.generated_image_label.setPixmap(pix_map)

    def run(self):
        self.output_tensor = self.dynamic_model.forward_and_recover(self.input_tensor, alpha=self.alpha_0)

    def on_save_image_click(self):
        self.save_image()

    def on_change_attributes_click(self):
        self.widget = ChooseAttributesWidget(main_widget=self)
        self.widget.show()

    def show_8_check_box_state_changed(self):
        if self.show_8_check_box.isChecked():
            self.show_16_check_box.setChecked(False)
            self.show_24_check_box.setChecked(False)
        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    def show_16_check_box_state_changed(self):
        if self.show_16_check_box.isChecked():
            self.show_8_check_box.setChecked(False)
            self.show_24_check_box.setChecked(False)
        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    def show_24_check_box_state_changed(self):
        if self.show_24_check_box.isChecked():
            self.show_8_check_box.setChecked(False)
            self.show_16_check_box.setChecked(False)
        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    def save_image(self):
        save_path = os.path.join('results', 'gui_results')
        if not os.path.exists(save_path):
            save_path = os.path.join('..', save_path)
        save_path = os.path.join(save_path, '%04d.png' % self.saved_images_count)
        self.output_image.save(save_path)
        print('image saved to %s' % save_path)
        self.saved_images_count += 1
        return save_path

    def choose_network(self):
        self.widget = ChooseAttributesWidget(main_widget=self)
        self.widget.show()

    def on_generation_button_click(self):
        self.generation_noise = self.generate()
        self.input_tensor = self.generation_noise
        self.run()
        self.set_output_image()

if __name__ == '__main__':
    # Initialise the application
    app = QApplication([])
    # Call the main widget
    ex = MainDynamicDcGanWidget()
    app.exec_()

