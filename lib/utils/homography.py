"""
Random homography generation.

- sample_homography
- find_map_and_mask
"""
import numpy as np
import cv2
from scipy.stats import truncnorm

def refine_homography(H_left, H_right, h, w):
    """
    To perform H refinement such that they are exact.
    The idea is to find three pairs of corners such that they are all integers,
    and then to estimate a homography from these corners.
    """
    center = np.array([
        [0, 0], [w, 0],
        [0, h], [w, h]
    ]).astype(np.float32)
    
    # project
    left = cv2.perspectiveTransform(center.reshape(4, -1, 2), H_left)
    # round, the key step
    left = left.astype(int).astype(np.float32).reshape(4, 2)
    
    # project
    right = cv2.perspectiveTransform(center.reshape(4, -1, 2), H_right)
    # round, the key step
    right = right.astype(int).astype(np.float32).reshape(4, 2)
    
    # find refined homography
    H_left = cv2.getPerspectiveTransform(center, left)
    H_right = cv2.getPerspectiveTransform(center, right)
    
    H = cv2.getPerspectiveTransform(left, right)
    
    return H_left, H_right, H
    

def sample_homography(h, w, **kargs):
    """
    Generate a random homography.
    :param h, w: source image height and width
    :param kargs:
        scale_ratio: (low, high)
        perspective_ratio: (low, high)
        max_angle: in radian
        direction: either 'left' or 'right'
    :return: a homography matrix H
    """
    scale_ratio = kargs.get('scale_ratio', (0.9, 1))
    perspective_ratio = kargs.get('perspective_ratio', (0.6, 0.6))
    max_angle = kargs.get('max_angle', np.pi / 12)
    direction = kargs.get('direction')
    
    H1 = sample_scale(h, w, scale_ratio)
    H2 = sample_rotation(h, w, max_angle)
    H3 = sample_perspective(h, w, direction, perspective_ratio)
    
    H = H1.dot(H2).dot(H3)
    
    return H
    

def sample_scale(h, w, scale_ratio=(0.6, 0.9)):
    """
    Random scale transformation
    :param h, w: original image height and width
    :param direction: either 'left' or 'right'.
    :param scale_ratio: (low, high)
    :return: H, (3, 3)
    """
    
    # sample a single ratio
    ratio = np.random.uniform(*scale_ratio)

    h_margin = (h - h * ratio) / 2
    w_margin = (w - w * ratio) / 2
    dst_corners = np.array([[0, 0], [0, h], [w, h], [w, 0]]).astype(np.float32)
    src_corners = np.array([
        [w_margin, h_margin],
        [w_margin, h - h_margin],
        [w - w_margin, h - h_margin],
        [w - w_margin, h_margin]]).astype(np.float32)
    
    # random x, y translation
    x_delta = np.random.uniform(-w_margin, w_margin)
    y_delta = np.random.uniform(-h_margin, h_margin)
    delta = np.array([[x_delta, y_delta]]).astype(np.float32)
    
    src_corners += delta

    return cv2.getPerspectiveTransform(src_corners, dst_corners)

def sample_perspective(h, w, direction, perspective_ratio=(0.5, 0.9)):
    """
    Random perspective transformation
    :param h, w: original image height and width
    :param direction: either 'left' or 'right'.
    :param perspective_ratio: (low, high) the ratio at which the edge will be cut.
    :return: H, (3, 3)
    """
    # get (low, high)
    # sample a single ratio
    ratio = np.random.uniform(*perspective_ratio)
    
    margin = (h - h * ratio) / 2
    dst_corners = np.array([[0, 0], [0, h], [w, h], [w, 0]]).astype(np.float32)
    if direction == 'left':
        src_corners = np.array([[0, margin], [0, h - margin], [w, h], [w, 0]]).astype(np.float32)
    elif direction == 'right':
        src_corners = np.array([[0, 0], [0, h], [w, h - margin], [w, margin]]).astype(np.float32)
    
    return cv2.getPerspectiveTransform(src_corners, dst_corners)

def sample_rotation(h, w, max_angle=np.pi / 6):
    """
    Random rotation transformation
    :param h, w: image size
    :param max_angle: maximum angle variation
    """

    dst_corners = np.array([[0, 0], [0, h], [w, h], [w, 0]]).astype(np.float32)
    src_corners = np.array([[0, 0], [0, h], [w, h], [w, 0]]).astype(np.float32)
    # angle = truncnorm.rvs(-max_angle / 2, max_angle / 2, loc=0., scale=max_angle/2.)
    angle = np.random.uniform(-max_angle / 2, max_angle / 2)
    center = np.mean(src_corners, axis=0)
    rot_mat = np.array(
        [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle), np.cos(angle)]]
    )
    # rotate around the center
    src_corners = np.matmul(src_corners - center, rot_mat)
    
    # rescale to within range
    w_range = np.max(src_corners[:, 0]) * 2
    h_range = np.max(src_corners[:, 1]) * 2
    scale_factor = min(w / w_range, h / h_range)
    src_corners = src_corners * scale_factor + center
    src_corners = src_corners.astype(np.float32)
    
    return cv2.getPerspectiveTransform(src_corners, dst_corners)

def find_map_and_mask(h0, w0, h1, w1, H):
    """
    Get correspondence map and mask from homography
    :param h0, w0, h1, h1: image size
    :param H: (3, 3) homography
    :return:
        map: corrsepondence map (h0, w0, 2), int, each being (x, y)
        mask: valid mask (h0, w0), boolean
    """
    
    # xs, ys: (h0, w0)
    xs, ys = np.meshgrid(np.arange(w0), np.arange(h0))
    # xy: (h0, w0, 2)
    xy = np.stack([xs, ys], axis=2)
    # xy: (h0 * w0, 1, 2), for cv2.perspectiveTransform
    xy = xy.reshape(-1, 1, 2).astype(np.float32)
    # get map, (h0 * w0, 1, 2)
    flow_map = cv2.perspectiveTransform(xy, H).astype(np.int)
    # reshape to (h0, w0, 2)
    flow_map = flow_map.reshape(h0, w0, 2)
    
    # get valid mask
    mask = ((0 <= flow_map[:, :, 0]) & (flow_map[:, :, 0] < w1) &
            (0 <= flow_map[:, :, 1]) & (flow_map[:, :, 1] < h1))
    
    return flow_map, mask
    


def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        help='image file'
    )
    args = parser.parse_args()
    
    file = args.file
    
    from skimage import io
    import matplotlib.pyplot as plt
    
    img = io.imread(file)
    h, w = img.shape[:2]
    # H = sample_perspective(h, w, 'left')
    while True:
        H = sample_homography(h, w, direction='left')
        new = cv2.warpPerspective(img, H, dsize=(w, h))
        plt.subplot(1, 2, 1)
        plt.imshow(new)
        H = sample_homography(h, w, direction='right')
        new = cv2.warpPerspective(img, H, dsize=(w, h))
        plt.subplot(1, 2, 2)
        plt.imshow(new)
        plt.show()
    

if __name__ == '__main__':
    test()

    

