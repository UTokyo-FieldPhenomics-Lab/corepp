from torchvision.transforms import v2
import numpy as np
import numbers
from PIL import Image
import matplotlib.pyplot as plt


def imshow(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def get_padding(image, size):    
    w, h = image.size
    max_wh = size
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class Pad(object):
    def __init__(self, size=512, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """

        img = Image.fromarray(img)
        return v2.functional.pad(img, get_padding(img, self.size), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)
    

class Rotate(object):
    def __init__(self, angle=45):
        self.angle = angle

    def __call__(self, img):
        angle = np.random.randint(0, self.angle)
        return v2.functional.rotate(img, angle)
    

class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return v2.functional.horizontal_flip(img)  
        

class RandomVerticalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return v2.functional.vertical_flip(img)
        

class AdjustBrightness(object):
    def __init__(self, brightness_factor=0.2):
        self.bf = brightness_factor

    def __call__(self, img):
        return v2.functional.adjust_brightness(img, self.bf)