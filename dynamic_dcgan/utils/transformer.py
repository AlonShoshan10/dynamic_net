import torchvision.transforms as transforms
from PySide2 import QtGui
import sys
sys.modules['PyQt5.QtGui'] = QtGui
from PIL import Image, ImageQt


class Transformer():
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.to_grey = transforms.Grayscale()
        self.to_pil_image = transforms.ToPILImage()

    def pil2pixmap(self, im):
        im = ImageQt.ImageQt(im)
        return QtGui.QPixmap.fromImage(im)

    def combine_pil_images(self, im1, im2, hight=128, width=64, pixmap=True):
        factor_1 = im1.width / hight
        factor_2 = im2.width / hight
        im1 = im1.resize((int(im1.width / factor_1), hight), Image.ANTIALIAS)
        im2 = im2.resize((int(im2.width / factor_2), hight), Image.ANTIALIAS)
        new_im = Image.new(im1.mode, (hight, hight))
        new_im.paste(im1, (0, 0))
        new_im.paste(im2, (width, 0))
        if pixmap:
            return self.pil2pixmap(new_im)
        else:
            return new_im

    def resize_to_max(self, im, max_size):
        if im.height > im.width:
            factor = max_size / im.height
        else:
            factor = max_size / im.width
        return im.resize((int(im.width * factor), int(im.height * factor)), Image.ANTIALIAS)






