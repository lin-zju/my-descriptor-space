"""
Keypoint related utilities.

- sample_uniform_keypoints
- draw_kps
"""

from .homography import find_map_and_mask
import numpy as np
import cv2

def sample_harris_keypoints(img0, img1, H, num_kps):
    """
    Sample ground truth correspondences
    Only keypoints common in the two images will be chosen.
    
    :param img: numpy, (H, W, 3), uint8
    :param H: (3, 3) homography matrix
    :param num_kps: number of keypoints to sample
    
    :return kps_left, kps_right:
        kps_left: numpy, (N, 2), float, each being (x, y)
        kps_right: numpy, (N, 2), float, each being (x, y)
    """
    
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    # get valid mask
    map, mask = find_map_and_mask(h0, w0, h1, w1, H)
    return _harris_sample(img0, map, mask, num_kps)

def sample_uniform_keypoints(img0, img1, H, num_kps=3000):
    """
    Sample ground truth correspondences
    Only keypoints present in both images will be sampled.
    
    
    :param img: numpy, (H, W, 3), uint8
    :param H: (3, 3) homography matrix
    :param num_kps: number of keypoints to sample
    
    :return kps_left, kps_right:
        kps_left: numpy, (N, 2), float, each being (x, y)
        kps_right: numpy, (N, 2), float, each being (x, y)
    """
    
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    # get valid mask
    map, mask = find_map_and_mask(h0, w0, h1, w1, H)
    return _uniform_sample(map, mask, num_kps)
    
def _uniform_sample(map, mask, num_kps):
    """
    
    :param map: numpy, (H, W, 2), int
    :param mask: (H, W) boolean
    :param num_kps: number of keypoints
    :return:
    """
    # valid_kps: (M, 2), all valid keypoints
    valid_kps = np.argwhere(mask)
    # to (x, y)
    valid_kps = valid_kps[:, ::-1]
    
    # random choice
    indices = np.arange(len(valid_kps))
    indices = np.random.choice(indices, num_kps, replace=False)
    
    # (N, 2)
    kps_src = valid_kps[indices]
    
    xs = kps_src[:, 0]
    ys = kps_src[:, 1]
    
    # get targets
    # (N, 2)
    kps_tgt = map[ys, xs]
    
    return kps_src, kps_tgt


def _harris_sample(img, map, mask, num_kps):
    """
    Sample keypoints from high-texture areas.
    
    :param img: left image
    :param map: numpy, (H, W, 2), int
    :param mask: (H, W) boolean
    :param num_kps: number of keypoints
    :return:
    """
    
    # using harris corner detection, get the mask of high texture area
    harris_img = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), 2, 3, 0.04)
    harris_msk = harris_img > np.percentile(harris_img.flatten(), 90)
    
    # get high texture mask
    mask = mask & harris_msk
    
    # valid_kps: (M, 2), all valid keypoints
    valid_kps = np.argwhere(mask)
    # to (x, y)
    valid_kps = valid_kps[:, ::-1]
    
    # random choice
    indices = np.arange(len(valid_kps))
    if len(indices) > num_kps:
        indices = np.random.choice(indices, num_kps, replace=False)
    
    # (N, 2)
    kps_src = valid_kps[indices]
    
    xs = kps_src[:, 0]
    ys = kps_src[:, 1]
    
    # get targets
    # (N, 2)
    kps_tgt = map[ys, xs]
    
    return kps_src, kps_tgt
    
    
    
    
    
    
    
    
    
    
    
    

