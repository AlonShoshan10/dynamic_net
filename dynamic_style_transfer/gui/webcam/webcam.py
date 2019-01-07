import cv2
import torchvision.transforms as transforms
from PIL import Image


class Webcam():
    def __init__(self):
        pass

    def take_photo(self):
        camera = cv2.VideoCapture(0)
        winname = 'Photo'
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 600, 100)
        while True:
            return_value, image = camera.read()
            cv2.imshow(winname, image)
            key = cv2.waitKey(50)
            if key in [32, 13]:
                break
        camera.release()
        del(camera)
        cv2.destroyAllWindows()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = image.shape
        image = transforms.ToPILImage()(image)
        image = image.resize((int(shape[1]*1.4), int(shape[0]*1.4)), Image.ANTIALIAS)
        return image