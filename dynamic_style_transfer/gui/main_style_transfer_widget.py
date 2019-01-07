from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QLabel, QApplication, QPushButton, QInputDialog, QLayout
from PySide2 import QtGui, QtCore
from gui.base_widget import BaseWidget
import os
import torch
from gui.choose_style_widget import ChooseStyleWidget
import PIL.Image as Image


class MainStyleTransferWidget(BaseWidget):
    def __init__(self, parent=None, app_name='Style Transfer'):
        super(MainStyleTransferWidget, self).__init__(parent=parent, widget_name=app_name)
        self.output_image_max_height = 640
        self.output_image_max_width = 768
        self.dual_mode = False
        self.choose_network()
        # make self.input_and_style_layout
        self.input_and_style_layout = QHBoxLayout()
        # make main layout
        self.main_layout = QHBoxLayout()
        self.output_image_label = QLabel(self)
        output_image_layout = self.make_image_layout(self.output_image_label, 'Output Image')
        self.output_image_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.main_layout.addLayout(output_image_layout)
        # make sliders
        self.multi_alpha = False
        self.alpha_0 = 0
        self.alpha_1 = 0
        self.alpha_2 = 0
        self.slider_layout = QVBoxLayout()
        self.make_sliders()
        # make check boxes
        check_boxes_layout = QHBoxLayout()
        self.multi_alpha_check_box = QCheckBox("Multi Blocks")
        self.multi_alpha_check_box.stateChanged.connect(self.multi_alpha_check_box_state_changed)
        self.show_input_and_style_check_box = QCheckBox("Show Input and Style")
        self.show_input_and_style_check_box.stateChanged.connect(self.show_input_and_style_check_box_state_changed)
        check_boxes_layout.addWidget(self.multi_alpha_check_box)
        check_boxes_layout.addWidget(self.show_input_and_style_check_box)
        # make save image Button
        save_image_button = QPushButton('Save Image')
        save_image_button.clicked.connect(self.on_save_image_click)
        change_style_button = QPushButton('Change Style')
        change_style_button.clicked.connect(self.on_change_style_click)
        # make buttons layout
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(save_image_button)
        self.buttons_layout.addWidget(change_style_button)
        # widget layout
        self.layout = QVBoxLayout()
        self.layout.addLayout(self.input_and_style_layout)
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
        self.input_image = None
        self.style_image = None
        self.output_image = None
        self.saved_images_count = 0
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
        if self.multi_alpha:
            alpha_0_slider_layout, self.alpha_0_slider, self.alpha_0_txt = self.make_slider_layout(self.alpha_0_slider_changed, val=self.alpha_0)
            self.alpha_0_txt.setText(U"\U0001D6C2 0 = %.3f" % self.alpha_0)
            alpha_1_slider_layout, self.alpha_1_slider, self.alpha_1_txt = self.make_slider_layout(self.alpha_1_slider_changed, val=self.alpha_1)
            self.alpha_1_txt.setText(U"\U0001D6C2 1 = %.3f" % self.alpha_1)
            alpha_2_slider_layout, self.alpha_2_slider, self.alpha_2_txt = self.make_slider_layout(self.alpha_2_slider_changed, val=self.alpha_2)
            self.alpha_2_txt.setText(U"\U0001D6C2 2 = %.3f" % self.alpha_2)
            self.slider_layout.addLayout(alpha_0_slider_layout)
            self.slider_layout.addLayout(alpha_1_slider_layout)
            self.slider_layout.addLayout(alpha_2_slider_layout)
        else:
            alpha_slider_layout, self.alpha_slider, self.alpha_txt = self.make_slider_layout(self.alpha_slider_changed, val=self.alpha_0)
            self.alpha_txt.setText(U"\U0001D6C2 = %.3f" % self.alpha_0)
            self.slider_layout.addLayout(alpha_slider_layout)

    def alpha_0_slider_changed(self):
        self.alpha_0 = (self.alpha_0_slider.value() / 100) ** 1
        if self.alpha_0 > 0.96:
            self.alpha_0 = 1
        self.alpha_0_txt.setText(U"\U0001D6C2 0 = %.3f" % self.alpha_0)
        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    def alpha_1_slider_changed(self):
        self.alpha_1 = (self.alpha_1_slider.value() / 100) ** 1
        if self.alpha_1 > 0.96:
            self.alpha_1 = 1
        self.alpha_1_txt.setText(U"\U0001D6C2 1 = %.3f" % self.alpha_1)
        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    def alpha_2_slider_changed(self):
        self.alpha_2 = (self.alpha_2_slider.value() / 100) ** 1
        if self.alpha_2 > 0.96:
            self.alpha_2 = 1
        self.alpha_2_txt.setText(U"\U0001D6C2 2 = %.3f" % self.alpha_2)
        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    def alpha_slider_changed(self):
        self.alpha_0 = (self.alpha_slider.value() / 100) ** 1
        if self.alpha_0 > 0.96:
            self.alpha_0 = 1
        self.alpha_1 = self.alpha_0
        self.alpha_2 = self.alpha_0
        self.alpha_txt.setText(U"\U0001D6C2 = %.3f" % self.alpha_0)
        if self.input_tensor is not None:
            self.run()
            self.set_output_image()

    # check box changed
    def multi_alpha_check_box_state_changed(self):
        if self.multi_alpha_check_box.isChecked():
            self.multi_alpha = True
            self.clear_layout(self.slider_layout)
            self.make_sliders()
        else:
            self.multi_alpha = False
            self.clear_layout(self.slider_layout)
            self.make_sliders()
            self.alpha_slider_changed()
        self.show()

    def show_input_and_style_check_box_state_changed(self):
        if self.show_input_and_style_check_box.isChecked():
            self.make_input_and_style_layout()
            self.set_style_image()
            self.set_input_image()
        else:
            self.clear_layout(self.input_and_style_layout)
        self.show()

    # drop event
    def execute_drop_event(self, file_name):
        self.input_image = Image.open(file_name)
        if self.show_input_and_style_check_box.isChecked():
            self.set_input_image()
        if self.dynamic_model is not None:
            self.calc_input_tensor()
            self.run()
            self.set_output_image()

    # images functions
    def set_input_image(self):
        if self.input_image is not None:
            self.input_image_label.setPixmap(self.make_pix_map(self.input_image, max_size=128))

    def set_style_image(self):
        if self.show_input_and_style_check_box.isChecked() and self.style_image is not None:
            self.style_image_label.setPixmap(self.make_pix_map(self.style_image, max_size=128))

    def set_output_image(self):
        self.output_image = self.transformer.to_pil_image(self.output_tensor.clamp(min=0.0, max=1).cpu().squeeze(dim=0))
        if self.output_image.height > self.output_image_max_height:
            pix_map = self.make_pix_map(self.output_image, max_size=self.output_image_max_height)
        elif self.output_image.width > self.output_image_max_width:
            pix_map = self.make_pix_map(self.output_image, max_size=self.output_image_max_width)
        else:
            pix_map = self.make_pix_map(self.output_image)
        self.output_image_label.setPixmap(pix_map)

    def calc_input_tensor(self):
        self.input_tensor = self.transformer.to_tensor(self.input_image).to(self.dynamic_model.device)
        self.input_tensor = self.dynamic_model.normalize(self.input_tensor)
        self.input_tensor = self.input_tensor.expand(1, -1, -1, -1)
        if self.input_tensor.shape[2] % 2 is not 0:
            self.input_tensor = self.input_tensor[:, :, 0:-1, :]
        if self.input_tensor.shape[2] % 4 is not 0:
            self.input_tensor = self.input_tensor[:, :, 1:-1, :]
        if self.input_tensor.shape[3] % 2 is not 0:
            self.input_tensor = self.input_tensor[:, :, :, 0:-1]
        if self.input_tensor.shape[3] % 4 is not 0:
            self.input_tensor = self.input_tensor[:, :, :, 1:-1]

    def run(self):
        if self.multi_alpha:
            self.output_tensor = self.dynamic_model.forward_and_recover(self.input_tensor, alpha_0=self.alpha_0, alpha_1=self.alpha_1, alpha_2=self.alpha_2)
        else:
            self.output_tensor = self.dynamic_model.forward_and_recover(self.input_tensor, alpha_0=self.alpha_0)

    def on_save_image_click(self):
        self.save_image()

    def on_change_style_click(self):
        self.choose_network()

    def choose_network(self):
        self.widget = ChooseStyleWidget(main_widget=self)
        self.widget.show()

    def save_image(self):
        save_path = os.path.join('results', 'gui_results')
        if not os.path.exists(save_path):
            save_path = os.path.join('..', save_path)
        save_path = os.path.join(save_path, '%04d.png' % self.saved_images_count)
        self.output_image.save(save_path)
        print('image saved to %s' % save_path)
        self.saved_images_count += 1
        return save_path

if __name__ == '__main__':
    # Initialise the application
    app = QApplication([])
    # Call the main widget
    ex = MainStyleTransferWidget()
    app.exec_()

