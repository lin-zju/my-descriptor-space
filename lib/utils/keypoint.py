"""
Keypoint related utilities.

- sample_uniform_keypoints
- draw_kps
"""

from .homography import find_map_and_mask
import numpy as np
import cv2

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
    
    
    
    
    
    
    

