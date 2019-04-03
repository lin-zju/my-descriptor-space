import torch
import numpy as np

def tonumpyimg(img):
    """
    Convert a normalized tensor image to unnormalized uint8 numpy image
    For single channel image, no unnormalization is done.
    
    :param img: torch, normalized, (3, H, W), (H, W)
    :return: numpy: (H, W, 3), (H, W). uint8
    """
    
    return touint8(tonumpy(unnormalize_torch(img)))

def tonumpy(img):
    """
    Convert torch image map to numpy image map
    Note the range is not change
    
    :param img: tensor, shape (C, H, W), (H, W)
    :return: numpy, shape (H, W, C), (H, W)
    """
    if len(img.size()) == 2:
        return img.cpu().detach().numpy()
    
    return img.permute(1, 2, 0).cpu().detach().numpy()

def touint8(img):
    """
    Convert float numpy image to uint8 image
    :param img: numpy image, float, (0, 1)
    :return: uint8 image
    """
    img = img * 255
    return img.astype(np.uint8)

def normalize_torch(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize a torch image.
    :param img: (3, H, W), in range (0, 1)
    """
    img = img.clone()
    img -= torch.tensor(mean).view(3, 1, 1)
    img /= torch.tensor(std).view(3, 1, 1)
    
    return img

def unnormalize_torch(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a normalized Tensor image to unnormalized form
    For single channel image, no normalization is done.
    :param img: (C, H, W), (H, W)
    """
    if img.size()[0] == 3:
        img = img.clone()
        img *= torch.Tensor(std).view(3, 1, 1)
        img += torch.Tensor(mean).view(3, 1, 1)
    
    return img

def gray2RGB(img_raw):
    """
    Convert a gray image to RGB
    :param img_raw: (H, W, 3) or (H, W), uint8, numpy
    :return: (H, W, 3)
    """
    if len(img_raw.shape) == 2:
        img_raw = np.repeat(img_raw[:, :, None], 3, axis=2)
    if img_raw.shape[2] > 3:
        img_raw = img_raw[:, :, :3]
    return img_raw
