import torch.nn.functional as F
import cv2
import torch

def resize_max_length(img, length):
    """
    Resize an image such that its long edge does not exceed max length.
    :param img: (H, W, 3), numpy
    :param length: int, length of the maximum edge.
    :return:
    """
    h, w = img.shape[:2]
    ratio = length / max(h, w)
    h, w = int(round(h * ratio)), int(round(w * ratio))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def sample_descriptor(descs, kps, images):
    """
    Sample descritpors without upsampling.

    descs: [B, D, H', W']
    kps: [B, N, 2], kps are the original pixels of images
    images: [B, D, H, W]
    :return descrs [B, N, D]
    """
    h, w = images.shape[2:]
    with torch.no_grad():
        kps = kps.clone().detach()
        kps[:, :, 0] = (kps[:, :, 0] / (float(w) / 2.)) - 1.
        kps[:, :, 1] = (kps[:, :, 1] / (float(h) / 2.)) - 1.
        kps = kps.unsqueeze(dim=1)
    descs = F.grid_sample(descs, kps)
    descs = descs[:, :, 0, :].permute(0, 2, 1)
    
    return F.normalize(descs, p=2, dim=2)


def flatten_image(img):
    """
    Flatten an image

    :param img: numpy, shape (H, W, C)
    :return: (H * W, C)
    """
    H, W, C = img.shape
    return img.reshape(H * W, C)


def unflatten_image(img, size):
    """
    Unflatten an image

    :param img: numpy, shape (H*W, C)
    :param size: (w, h)
    :return: shape (H, W, C)
    """
    w, h = size
    assert w * h == img.shape[0], 'In unflatten_image: shape mismatch'
    
    return img.reshape(h, w, -1)



