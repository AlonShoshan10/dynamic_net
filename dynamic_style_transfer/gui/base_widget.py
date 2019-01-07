from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QSlider, QLabel, QApplication, QCheckBox, QMainWindow
from PySide2 import QtGui, QtCore
from skimage import io
from PIL import Image
from utils.transformer import Transformer
import torch


class BaseWidget(QWidget):
    def __init__(self, parent=None, widget_name='Base Widget', geometry=None):
        super(BaseWidget, self).__init__(parent)
        self.slider_range = (0, 100)
        self.widget_name = widget_name
        # main window
        if geometry is None:
            self.setGeometry(700, 30, 300, 560)
        else:
            self.setGeometry(geometry[0], geometry[1], geometry[2], geometry[3])
        self.setWindowTitle(widget_name)
        # enable dragging and dropping onto the GUI
        self.setAcceptDrops(True)
        self.drop_event_file_name = None
        self.transformer = Transformer()

    def make_image_layout(self, image_ql, title):
        txt = QLabel(title)
        image_layout = QVBoxLayout()
        image_layout.addWidget(txt)
        image_layout.addWidget(image_ql)
        txt.setAlignment(QtCore.Qt.AlignHCenter)
        image_ql.setAlignment(QtCore.Qt.AlignHCenter)
        return image_layout

    def make_slider_layout(self, value_changed_func, val=0):
        slider = QSlider(QtCore.Qt.Horizontal, self)
        slider.setMinimum(self.slider_range[0])
        slider.setMaximum(self.slider_range[1])
        slider.setValue(val * 100)
        slider.valueChanged[int].connect(value_changed_func)
        layout_slider = QVBoxLayout()
        layout_slider.addWidget(slider)
        txt = QLabel()
        layout_slider.addWidget(txt)
        txt.setAlignment(QtCore.Qt.AlignHCenter)
        return layout_slider, slider, txt

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        if e.mimeData().hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()
            # Workaround for OSx dragging and dropping
            for url in e.mimeData().urls():
                file_name = str(url.toLocalFile())
            self.drop_event_file_name = file_name
            self.execute_drop_event(file_name)
        else:
            e.ignore()

    def execute_drop_event(self, file_name):
        pass

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

    def load_pix_map(self, path, max_size=None):
        loaded_image = Image.open(path)
        return self.make_pix_map(loaded_image, max_size=max_size)

    def make_pix_map(self, image, max_size=None):
        if max_size is not None:
            if image.height > image.width:
                factor = max_size / image.height
            else:
                factor = max_size / image.width
            image = image.resize((int(image.width * factor), int(image.height * factor)), Image.ANTIALIAS)
        return self.transformer.pil2pixmap(image)




if __name__ == '__main__':
    # Initialise the application
    app = QApplication([])
    # Call the main widget
    ex = BaseWidget()
    app.exec_()
