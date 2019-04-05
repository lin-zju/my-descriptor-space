"""
Microsoft COCO dataset adapted for descriptor training. Random homography is
applied to every training example.

The images are resized to 240x320

"""
import torch
import random
import glob
import os
import numpy as np
from skimage import io
import cv2
from torch.utils.data import Dataset
from lib.utils.homography import sample_homography, find_map_and_mask, refine_homography
from lib.utils.keypoint import sample_uniform_keypoints, sample_harris_keypoints
from lib.utils.convert import normalize_torch, gray2RGB
from lib.utils.misc import resize_max_length


class Hpatches(Dataset):
    def __init__(self, root, mode, transforms=None, size=640, num_kps=1000):
        """
        :param mode: either 'test' or 'val'
        """
        Dataset.__init__(self)
        self.img_paths = self.get_img_paths(root)
        self.transforms = transforms
        self.size = size
        self.num_kps = num_kps
        self.mode = mode
    
    def __getitem__(self, index):
        """
        :return:
            data:
                img0: (3, H, W), float
                img1: (3, H, W), float
            targets:
                kps0: (N, 2), long Tensor
                kps1: (N, 2), long Tensor
                map: (H, W, 2), long Tensor
                mask: (H, W), byte Tensor
                H: (3, 3), numpy, float
        """
        img0_path, img1_path, H_path = self.img_paths[index]
        
        # load images
        img0, scale_h0, scale_w0 = self.read_img(img0_path, self.size)
        img1, scale_h1, scale_w1 = self.read_img(img1_path, self.size)

        # load H. Note scale changes might be applied
        scale_ratio = scale_h0, scale_w0, scale_h1, scale_w1
        H = self.read_H(H_path, scale_ratio)
        
        # find flow map and mask
        # map: int, mask: boolean
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        map, mask = find_map_and_mask(h0, w0, h1, w1, H)
        
        # find keypoints
        # kps: int, (N, 2)
        kps0, kps1 = sample_harris_keypoints(img0, img1, H, self.num_kps)
        
        # we should assume that only data augmentation are applied
        if self.transforms:
            img0 = self.transforms(img0)
            img1 = self.transforms(img1)
        
        # to tensor
        img0 = torch.from_numpy(img0).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        
        # to range (0, 1)
        img0 = img0 / 255.0
        img1 = img1 / 255.0

        # normalize
        img0 = normalize_torch(img0)
        img1 = normalize_torch(img1)
        
        # keypoints
        kps0 = torch.from_numpy(kps0).float()
        kps1 = torch.from_numpy(kps1).float()
        map = torch.from_numpy(map).float()
        mask = torch.from_numpy(mask.astype(int)).byte()
        
        data = {
            'img0': img0,
            'img1': img1,
        }
        targets = {
            'kps0': kps0,
            'kps1': kps1,
            'map': map,
            'mask': mask,
            'H': H,
        }
        
        return data, targets
    
    def __len__(self):
        return len(self.img_paths)
    
    @staticmethod
    def get_img_paths(root):
        """
        Retrieve image paths for all images.
        """
        filelist = []
        for subdir in os.scandir(root):
            subdir = subdir.path
            ref = os.path.join(subdir, '1.ppm')
            for i in range(2, 6 + 1):
                H = os.path.join(subdir, 'H_1_{}'.format(i))
                other = os.path.join(subdir, '{}.ppm'.format(i))
                filelist.append((ref, other, H))
                
        return filelist
    
    @staticmethod
    def read_img(img_path, scale=None):
        """
        Read an image, and resize it
        :param img_path: path to image
        :param scale: either None, int or tuple. For tuple, this should be (w, h)
        :return:
            img: the resized image
            scale_h, scale_w: scale changes
        """
        img = gray2RGB(io.imread(img_path))
        oh, ow = img.shape[:2]
        if isinstance(scale, int):
            img = resize_max_length(img, scale)
        elif isinstance(scale, tuple):
            img = cv2.resize(img, scale)
        h, w = img.shape[:2]
        scale_h, scale_w = h / oh, w / ow
        return img, scale_h, scale_w
    
    @staticmethod
    def read_H(H_path, scale_ratio):
        """
        Read the homography
        :param H_path: the path
        :param scale_ratio:
            scale_h0, scale_w0, scale_h1, scale_w1.
            This will be used to rescale H.
        :return: H, numpy, (3, 3)
        """
        scale_h0, scale_w0, scale_h1, scale_w1 = scale_ratio
        H = np.loadtxt(H_path).astype(np.float32)
        H = np.diag([scale_w1, scale_h1, 1.0]).dot(H).dot(np.linalg.inv(np.diag([scale_w0, scale_h0, 1.0])))
        
        return H

class HpatchesViewpoint(Hpatches):
    def __init__(self, *args, **kargs):
        Hpatches.__init__(self, *args, **kargs)
        # only viewpoint changes
        # test and validation split
        # use 95 for validation and 200 for testing
        
        assert self.mode in ['val', 'test', 'all'], 'Invalid mode {} for hpatches viewpoint.'.format(self.mode)

        self.img_paths = sorted([x for x in self.img_paths if '/v' in x[0]])
        random.seed(233)
        random.shuffle(self.img_paths)
        if self.mode == 'val':
            self.img_paths = self.img_paths[:95]
        elif self.mode == 'test':
            self.img_paths = self.img_paths[95:]
        elif self.mode == 'all':
            pass
            
        

class HpatchesIllum(Hpatches):
    def __init__(self, *args, **kargs):
        Hpatches.__init__(self, *args, **kargs)
        # only illumination changes
        self.img_paths = [x for x in self.img_paths if '/i' in x[0]]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from lib.utils.convert import tonumpyimg, tonumpy
    from lib.utils.visualize import draw_match

    ds = Hpatches('data/HPATCHES')
    for data, targets in ds:
        left, right = data['img0'], data['img1']
        map, mask, H = targets['map'], targets['mask'], targets['H']
        kps0, kps1 = targets['kps0'], targets['kps1']
        left, right, mask = [tonumpyimg(x) for x in [left, right, mask]]
        h, w = left.shape[:2]
        warped = cv2.warpPerspective(left, H, (w, h))
        match = draw_match(left, kps0.numpy(), right, kps1.numpy())
        plt.imshow(match)
        
        
        plt.show()
