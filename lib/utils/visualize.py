"""
Visualization utilities
"""
import cv2
import numpy as np

import numpy as np
from sklearn.decomposition import PCA
import cv2
import torch

from lib.utils.misc import flatten_image, unflatten_image
from lib.utils.convert import torange, touint8, RGB2BGR
from lib.utils.convert import totensor, tonumpy, tofloat, unnormalize


def desc2RGB(desc):
    """
    Convert a descriptor map to RGB image.
    :param desc: shape (H, W, C), numpy
    :return: an image of shape (H, W, 3), range (0, 1)
    """
    reduced = reduce_desc_dim(desc)
    return torange(reduced, 0, 1)


def reduce_desc_dim(desc, dim=3):
    """
    Reduce feature map dimensions to dim
    :param desc: feature map of shape (H, W, C), numpy
    :param dim: the resulting dimension
    :return: dim-reduced feature map
    """
    
    pca = PCA(3)
    # first, flantten the map
    flattened = flatten_image(desc)
    # reduce dimension
    reduced = pca.fit_transform(flattened)
    h, w = desc.shape[:2]
    return unflatten_image(reduced, (w, h))


def draw_kps(img0, pix_pos0, img1, pix_pos1):
    """
    Draw keypoints, numpy version

    :param img0: (H, W, C)
    :param pix_pos0: (N, 2), (x, y)
    :param img1: (H, W, C)
    :param pix_pos1: (N, 2), (x, y)
    :return: a single image of keypoints
    """
    # img0, img1 = RGB2BGR(img0, img1)
    kps0 = [cv2.KeyPoint(pix[0], pix[1], 1) for pix in pix_pos0]
    img0 = cv2.drawKeypoints(img0, kps0, None)
    kps1 = [cv2.KeyPoint(pix[0], pix[1], 1) for pix in pix_pos1]
    img1 = cv2.drawKeypoints(img1, kps1, None)
    img = np.concatenate([img0, img1], axis=1)
    return img


def draw_kps_torch(img0, pix_pos0, img1, pix_pos1):
    """
    Draw keypoints, torch version
    :param img0: (C, H, W)
    :param pix_pos0: (N, 2), (x, y)
    :param img1:  (C, H, W)
    :param pix_pos1: (N, 2), (x, y)
    :return: a single torch image
    """
    img0, img1 = [tonumpy(x) for x in [img0, img1]]
    img0, img1 = [touint8(x) for x in [img0, img1]]
    pix_pos0, pix_pos1 = [x.numpy() for x in [pix_pos0, pix_pos1]]
    
    img = draw_kps(img0, pix_pos0, img1, pix_pos1)
    img = totensor(tofloat(img))
    return img


def draw_corr_torch(img0, pix_pos0, img1, pix_pos1):
    """
    Draw matches, torch version. Parameters are the same as draw keypoints
    """
    
    img0, img1 = [tonumpy(x) for x in [img0, img1]]
    img0, img1 = [touint8(x).astype(np.uint8) for x in [img0, img1]]
    pix_pos0, pix_pos1 = [x.numpy() for x in [pix_pos0, pix_pos1]]
    
    img = draw_corr(img0, pix_pos0, img1, pix_pos1)
    img = totensor(tofloat(img))
    return img


def draw_corr(img0, pix_pos0, img1, pix_pos1, num=None):
    """
    Draw matches, numpy version. Parameters are the same as draw keypoints
    """
    if num is None:
        inds = np.arange(len(pix_pos0))
        num = len(pix_pos0)
    else:
        inds = np.linspace(0, len(pix_pos0), num, endpoint=False).astype(np.int32)
    
    # img0, img1 = RGB2BGR(img0, img1)
    kps0 = [cv2.KeyPoint(pix[0], pix[1], 1) for pix in pix_pos0[inds]]
    kps1 = [cv2.KeyPoint(pix[0], pix[1], 1) for pix in pix_pos1[inds]]
    matches = [cv2.DMatch(i, i, 0) for i in range(num)]
    img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None)
    
    return img


def draw_img_desc_torch(img, desc):
    """
    img: [3, H, W]
    desc: [C, H, W]
    """
    img = (tonumpy(unnormalize(img)) * 255.).astype(np.uint8)  # [H, W, 3]
    desc = (desc2RGB(tonumpy(desc)) * 255.).astype(np.uint8)  # [H, W, 3]
    h, w = img.shape[:2]
    desc = cv2.resize(desc.copy(), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([img, desc], axis=1)


def draw_paired_img_desc_torch(img0, desc0, kps0, img1, desc1, kps1, H):
    """
    img0: [3, H_0, W_0]
    desc0: [C, H_0', W_0']
    kps0: [N, 2]
    img1: [3, H_1, W_1]
    desc1: [C, H_1', W_1']
    kps1: [N, 2]
    H: [3, 3]
    """
    img0 = (tonumpy(unnormalize(img0)) * 255.).astype(np.uint8)  # [H, W, 3]
    desc0 = (desc2RGB(tonumpy(desc0)) * 255.).astype(np.uint8)  # [H, W, 3]
    kps0 = kps0.detach().cpu().numpy()
    img1 = (tonumpy(unnormalize(img1)) * 255.).astype(np.uint8)
    desc1 = (desc2RGB(tonumpy(desc1)) * 255.).astype(np.uint8)
    kps1 = kps1.detach().cpu().numpy()
    
    # compute the corresponding region
    H = H.detach().cpu().numpy()
    h1, w1 = img1.shape[:2]
    pts1 = np.array(
        [[0, 0],
         [0, h1],
         [w1, h1],
         [w1, 0]]
    ).astype(np.float32)
    pts0 = cv2.perspectiveTransform(np.reshape(pts1, [1, -1, 2]), np.linalg.inv(H))[0]
    
    # draw the corresponding region on the image
    pts0 = pts0.astype(np.int32)
    img0 = cv2.polylines(img0.copy(), [pts0.reshape([-1, 2])], True, (255, 0, 0), thickness=5)
    desc0 = cv2.polylines(desc0.copy(), [pts0.reshape([-1, 2])], True, (255, 0, 0), thickness=5)
    # desc0 = cv2.warpPerspective(desc0.copy(), H, (w1, h1), flags=cv2.INTER_LINEAR)
    pts1 = pts1.astype(np.int32)
    img1 = cv2.polylines(img1.copy(), [pts1.reshape([-1, 2])], True, (255, 0, 0), thickness=5)
    desc1 = cv2.polylines(desc1.copy(), [pts1.reshape([-1, 2])], True, (255, 0, 0), thickness=5)
    
    # draw correspondences
    img = draw_corr(img0, kps0, img1, kps1, num=10)
    # desc = draw_corr(desc0, kps0, desc1, kps1, num=10)
    desc = draw_corr(desc0, kps1, desc1, kps1, num=10)
    return np.concatenate([img, desc], axis=0)


def draw_paired_desc_torch(desc0, kps0, image0, desc1, kps1, image1, H):
    """
    desc0: [C, H_0', W_0']
    kps0: [N, 2]
    image0: [3, H_0, W_0]
    desc1: [C, H_1', W_1']
    kps1: [N, 2]
    image1: [3, H_1, W_1]
    H: [3, 3]
    """
    desc0 = (desc2RGB(tonumpy(desc0)) * 255.).astype(np.uint8)  # [H, W, 3]
    kps0 = kps0.detach().cpu().numpy()
    desc1 = (desc2RGB(tonumpy(desc1)) * 255.).astype(np.uint8)
    kps1 = kps1.detach().cpu().numpy()
    
    # compute the corresponding region
    H = H.detach().cpu().numpy()
    h1, w1 = image1.shape[1:]
    pts1 = np.array(
        [[0, 0],
         [0, h1],
         [w1, h1],
         [w1, 0]]
    ).astype(np.float32)
    pts0 = cv2.perspectiveTransform(np.reshape(pts1, [1, -1, 2]), np.linalg.inv(H))[0]
    
    # compute the ratio between desc and image, and rescale the kps
    ratio_h0 = desc0.shape[0] / image0.shape[1]  # [H_0', W_0', 3], [3, H_0, W_0]
    ratio_w0 = desc0.shape[1] / image0.shape[2]
    pts0 *= np.array([[ratio_w0, ratio_h0]])
    ratio_h1 = desc1.shape[0] / image1.shape[1]
    ratio_w1 = desc1.shape[1] / image1.shape[2]
    pts1 *= np.array([[ratio_w1, ratio_h1]])
    
    # draw the corresponding region on the image
    pts0 = pts0.astype(np.int32)
    desc0 = cv2.polylines(desc0.copy(), [pts0.reshape([-1, 2])], True, (255, 0, 0), thickness=5)
    pts1 = pts1.astype(np.int32)
    desc1 = cv2.polylines(desc1.copy(), [pts1.reshape([-1, 2])], True, (255, 0, 0), thickness=5)
    
    # draw correspondences
    desc = draw_corr(desc0, kps0, desc1, kps1, num=10)
    desc = totensor(tofloat(desc))
    
    return desc


def draw_corspd_region_torch(img0, img1, H):
    """
    img0: [3, H_0, W_0]
    img1: [3, H_1, W_1]
    """
    img0 = (tonumpy(unnormalize(img0)) * 255.).astype(np.uint8)  # [H0, W0, 3]
    img1 = (tonumpy(unnormalize(img1)) * 255.).astype(np.uint8)  # [H1, W1, 3]
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()
    
    h1, w1 = img1.shape[:2]
    pts1 = np.array(
        [[0, 0],
         [0, h1],
         [w1, h1],
         [w1, 0]]
    ).astype(np.float32)
    pts0 = cv2.perspectiveTransform(np.reshape(pts1, [1, -1, 2]), np.linalg.inv(H))[0]
    
    # draw the corresponding region on the image
    pts0 = pts0.astype(np.int32)
    img0 = cv2.polylines(img0.copy(), [pts0.reshape([-1, 2])], True, (255, 0, 0), thickness=5)
    pts1 = pts1.astype(np.int32)
    img1 = cv2.polylines(img1.copy(), [pts1.reshape([-1, 2])], True, (255, 0, 0), thickness=5)
    
    return np.concatenate([img0, img1], axis=1)


def cls2RGB(cls, c):
    """
    cls: [C, H, W]
    c: number of classes
    """
    r = cls.clone()
    g = cls.clone()
    b = cls.clone()
    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70]
    ]
    for l in range(c):
        inds = (cls == l)
        r[inds] = colors[l][0]
        g[inds] = colors[l][1]
        b[inds] = colors[l][2]
    cls = torch.stack([r, g, b], dim=0)
    return cls

def draw_match(img0, kps0, img1, kps1, num_kps=20):
    """
    Draw matches in the two image
    :param img0, img1: (H, W, 3), numpy, uint8
    :param kps0, kps1: (N, 2), numpy, int, (x, y)
    :param num_kps: number of keypoints to draw
    :return: the image, (H, W, 3), numpy, uint8
    """
    
    indices = np.random.choice(range(len(kps0)), num_kps)
    kps0 = kps0[indices]
    kps1 = kps1[indices]
    
    # to opencv keypoints
    kps0 = [cv2.KeyPoint(xy[0], xy[1], 1) for xy in kps0]
    kps1 = [cv2.KeyPoint(xy[0], xy[1], 1) for xy in kps1]
    matches = [cv2.DMatch(i, i, 0) for i in range(num_kps)]
    img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None)
    
    return img


if __name__ == '__main__':
    desc = np.random.rand(256, 256, 128)
    img = desc2RGB(desc)
    print(img.dtype, img.shape, img.min(), img.max())


