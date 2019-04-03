# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import numpy as np
from torchvision.transforms import ColorJitter
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None):
        for t in self.transforms:
            img, kpts = t(img, kpts)
        if kpts is None:
            return img
        else:
            return img, kpts

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, img, kpts):
        return img / 255., kpts


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, kpts):
        img -= self.mean
        img /= self.std
        return img, kpts


class JpegCompress(object):
    def __init__(self, quality_low=15, quality_high=75):
        self.quality_low = quality_low
        self.quality_high = quality_high

    def __call__(self, img, kpts):
        if np.random.uniform(0, 1) < 0.5:
            return img, kpts

        quality = np.random.randint(self.quality_low, self.quality_high)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
        return img, kpts


class GaussianBlur(object):
    def __init__(self, blur_range=(3, 5, 7, 9, 11)):
        self.blur_range = blur_range

    def __call__(self, img, kpts):
        if np.random.uniform(0, 1) < 0.5:
            return img, kpts

        sigma = np.random.choice(self.blur_range, 1, p=(0.4, 0.3, 0.2, 0.05, 0.05))
        return cv2.GaussianBlur(img, (sigma, sigma), 0), kpts


class AddNoise(object):
    def __call__(self, img, kpts):
        if np.random.uniform(0, 1) < 0.66:
            return img, kpts

        # gaussian noise
        if np.random.uniform(0, 1) < 0.75:
            row, col, ch = img.shape
            mean = 0
            var = np.random.rand(1) * 0.3 * 256
            sigma = var**0.5
            gauss = sigma * np.random.randn(row,col) + mean
            gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
            img = img + gauss
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
        else:
            # motion blur
            sizes = [3, 5, 7, 9]
            size = sizes[int(np.random.randint(len(sizes), size=1))]
            kernel_motion_blur = np.zeros((size, size))
            if np.random.rand(1) < 0.5:
                kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
            else:
                kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size
            img = cv2.filter2D(img, -1, kernel_motion_blur)

        return img, kpts


class Jitter(object):
    def __init__(self, brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2):
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, kpts):
        if np.random.uniform(0, 1) < 0.66:
            return img, kpts

        img = np.asarray(self.jitter(Image.fromarray(img)))
        return img, kpts
