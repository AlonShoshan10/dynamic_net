from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QLabel, QApplication, QPushButton, QInputDialog
from PySide2 import QtGui, QtCore
from gui.main_style_transfer_widget import MainStyleTransferWidget
import os
import torch
from gui.choose_style_widget import ChooseStyleWidget
import PIL.Image as Image
from gui.webcam.webcam import Webcam


class WebcamStyleTransferWidget(MainStyleTransferWidget):
    def __init__(self, parent=None, app_name='Style Transfer'):
        super(WebcamStyleTransferWidget, self).__init__(parent=parent, app_name=app_name)
        self.webcam = Webcam()
        # make  Button
        take_photo_button = QPushButton('Take Photo')
        take_photo_button.clicked.connect(self.on_take_photo_click)
        self.buttons_layout.insertWidget(0, take_photo_button)
        self.show()


    def on_take_photo_click(self):
        self.input_image = self.webcam.take_photo()
        if self.show_input_and_style_check_box.isChecked():
            self.set_input_image()
        if self.dynamic_model is not None:
            self.calc_input_tensor()
            self.run()
            self.set_output_image()


if __name__ == '__main__':
    # Initialise the application
    app = QApplication([])
    # Call the main widget
    ex = WebcamStyleTransferWidget()
    app.exec_()

