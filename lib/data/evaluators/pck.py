import torch
from .base import Evaluator
from lib.utils.misc import sample_descriptor, compute_scale
import numpy as np
from lib.utils.nn_set2set_match.nn_set2set_match_layer import nn_set2set_match_cuda
import cv2

class DescPckEvaluator(Evaluator):
    def __init__(self, threshold):
        """
        Evaluate pck of your model.
        :param threshold: pixel threshold for PCK.
        """
        self.threshold = threshold
        self.pck = []
        
    def evaluate(self, data, targets, results):
        """
        Evaulate pck of descriptor outputs.
        Note we only support the B=1 case
        
        :param data: dict,
            img0, img1: images (B, 3, H, W)
        :param targets: dict
            kps0, kps1: keypoints (B, N, 2), original image scale
        :param results: dict
            descs0, descs1: descriptors
            Either (B, D, H, W), for ordinary method
            Or (B, 2, D, H, W)
        :return:
        """
        cpu = torch.device('cpu')
        image0 = data['img0'].to(cpu)
        image1 = data['img1'].to(cpu)
        kps0 = targets['kps0'].to(cpu)
        kps1 = targets['kps1'].to(cpu)
        descs0 = results['descs0'].to(cpu)
        descs1 = results['descs1'].to(cpu)

        if len(descs0.shape) == 5:
            # set to set match
            x1 = sample_descriptor(descs0[:, 0], kps0, image0)[0]
            x2 = sample_descriptor(descs0[:, 1], kps0, image0)[0]
            descs0 = torch.stack([x1, x2], dim=1)
    
            x1 = sample_descriptor(descs1[:, 0], kps1, image1)[0]
            x2 = sample_descriptor(descs1[:, 1], kps1, image1)[0]
            descs1 = torch.stack([x1, x2], dim=1)
        else:
            descs0 = sample_descriptor(descs0, kps0, image0)[0].numpy()
            descs1 = sample_descriptor(descs1, kps1, image1)[0].numpy()


        # self.scales.append(get_wrong_pixel_scale(descs0, kps0, descs1, kps1, image0, H))
        self.pck.append(compute_pck(descs0, kps0[0], descs1, kps1[0]))

    def average_precision(self):
        # print("pck: {}".format(np.mean(self.pck)))
        return np.mean(self.pck)
    
    def get_results(self):
        return np.mean(self.pck)
        

    def wrong_pixel_scale(self):
        if len(self.scales) == 0:
            return
        import matplotlib.pyplot as plt
        plt.hist(np.concatenate(self.scales), bins=100, range=(0, 2))
        plt.show()


def keep_valid_keypoints(keypoints, H, height, width):
    """
    Keep only keypoints that is present in the other image.

    :param keypoints: shape (N, 2)
    :param H: shape (3, 3)
    :param height, weight: target image height and width
    :return (keypoints, descriptors): valid ones
    """

    N = keypoints.shape[0]
    mapped = cv2.perspectiveTransform(keypoints.reshape(N, 1, 2).astype(np.float), H).reshape(N, 2)
    indices_valid = (
            (mapped[:, 0] >= 0) & (mapped[:, 0] < width) &
            (mapped[:, 1] >= 0) & (mapped[:, 1] < height)
    )

    return keypoints[indices_valid]

def nn_match(descs1, descs2):
    """
    Perform nearest neighbor match, using descriptors.

    This function uses OpenCV FlannBasedMatcher

    :param descs1: descriptors from image 1, (N1, D)
    :param descs2: descriptors from image 2, (N2, D)
    :return indices: indices into keypoints from image 2, (N1, D)
    """
    # diff = descs1[:, None, :] - descs2[None, :, :]
    # diff = np.linalg.norm(diff, ord=2, axis=2)
    # indices = np.argmin(diff, axis=1)

    flann = cv2.FlannBasedMatcher_create()
    matches = flann.match(descs1.astype(np.float32), descs2.astype(np.float32))
    indices = [x.trainIdx for x in matches]

    return indices

def nn_set2set_match(descs1, descs2):
    """
    Perform nearest neighbor match on CUDA, using sets of descriptors
    This function uses brute force

    descs1: [N1, 2, D]
    descs2: [N2, 2, D]
    indices: indices into keypoints from image 2, [N1, D]
    """
    idxs = nn_set2set_match_cuda(descs1.unsqueeze(0).cuda(), descs2.unsqueeze(0).cuda()).detach().cpu().long()
    return idxs[0]

def compute_matching_score(descrs0, kps0, descrs1, kps1, H, thresh=3):
    """
    Compute matching score given two sets of keypoints and descriptors.

    :param feats0, feats1: descriptors for the images, shape (N0, D)
    :param kps0, kps1: keypoints for the image, shape (N0, 2), each being (x, y)
    :param H: homography, shape (3, 3), from image 0 to image 1
    :param thresh: threshold in pixel
    :return: matchine score (%)
    """
    N0 = descrs0.shape[0]
    N1 = descrs1.shape[0]

    # matches points from image 0 to image 1, using NN
    idxs = nn_match(descrs0, descrs1)  # matched image 1 keypoint indices
    predicted = kps1[idxs]  # matched image 1 keypoints

    # ground truth matched location
    gt = cv2.perspectiveTransform(kps0.reshape(N0, 1, 2).astype(np.float), H).reshape(N0, 2)
    correct0 = np.linalg.norm(predicted - gt, 2, axis=1) <= thresh

    # 1 to 0
    idxs = nn_match(descrs1, descrs0)
    predicted = kps0[idxs]
    gt = cv2.perspectiveTransform(kps1.reshape(N1, 1, 2).astype(np.float), np.linalg.inv(H)).reshape(N1, 2)
    correct1 = np.linalg.norm(predicted - gt, 2, axis=1) <= thresh

    return (np.sum(correct1) + np.sum(correct0)) / (N1 + N0)

def sift_detector(img):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kps = sift.detect(img, None)
    kps_np = np.asarray([kp.pt for kp in kps])
    return np.round(kps_np).astype(np.int32)

def compute_pck(descs0, kps0, descs1, kps1, thresh=3):
    """
    Compute pck given two sets of keypoints and descriptors.

    :param descs0, descs1: descriptors for the images, shape (N0, D) descriptor, or (N0, 2, D) multi-scale descriptor
    :param kps0, kps1: keypoints for the image, shape (N0, 2), each being (x, y)
    :param thresh: threshold in pixel
    :return: matchine score (%)
    """
    # matches points from image 0 to image 1, using NN
    if len(descs0.shape) == 3:
        idxs = nn_set2set_match(descs0, descs1)
    else:
        idxs = nn_match(descs0, descs1)  # matched image 1 keypoint indices
    predicted = kps1[idxs]  # matched image 1 keypoints

    correct = np.linalg.norm(predicted - kps1, 2, axis=1) <= thresh

    return np.sum(correct) / len(correct)

def get_wrong_pixel_scale(descs0, kps0, descs1, kps1, image, H, thresh=3):
    """
    get the wrongly matched pixels of left image when matching it to right image
    """
    if len(descs0.shape) == 3:
        idxs = nn_set2set_match(descs0, descs1)
    else:
        idxs = nn_match(descs0, descs1)  # matched image 1 keypoint indices
    predicted = kps1[idxs]
    wrong = np.linalg.norm(predicted - kps1, 2, axis=1) > thresh
    # pixels = kps0.detach().cpu().long().numpy()[wrong]
    pixels = kps0.astype(np.int)[wrong]

    h, w = image.shape[1:]
    scale = compute_scale(H, h, w)
    scales = scale[pixels[:, 1], pixels[:, 0]]
    return scales, pixels
