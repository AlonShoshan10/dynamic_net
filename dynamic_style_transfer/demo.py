from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QLabel, QApplication, QPushButton, QInputDialog
from gui.webcam_style_transfer_widget import WebcamStyleTransferWidget

app = QApplication([])
ex = WebcamStyleTransferWidget()
app.exec_()

