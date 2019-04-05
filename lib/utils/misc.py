import torch.nn.functional as F
import cv2
import torch
import numpy as np

def homo_mm(H, x, y):
    [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = H
    C = m31 * x + m32 * y + m33
    x_prime = m11 * x + m12 * y + m13
    y_prime = m21 * x + m22 * y + m23
    return x_prime, y_prime, C

def compute_scale(H, h, w):
    """
    compute the pixel scale changes after homography transformation
    H: homography matrix, [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]]
    """
    [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = H
    yx = np.mgrid[:h, :w]
    x, y, C = homo_mm(H, yx[1], yx[0])
    # J: [2, 2, h, w]
    J = np.array([[m11 * C - m31 * x, m12 * C - m32 * x],
                 [m21 * C - m31 * y, m22 * C - m32 * y]])

    J /= C ** 2
    J = J.transpose(2, 3, 0, 1)
    # scale = np.sqrt(np.linalg.det(J)).astype(np.float32)
    scale = np.sqrt(np.abs(np.linalg.det(J))).astype(np.float32)
    return scale

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



