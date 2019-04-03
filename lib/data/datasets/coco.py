"""
Microsoft COCO dataset adapted for descriptor training. Random homography is
applied to every training example.

The images are resized to 240x320

"""
import torch
import glob
import os
import numpy as np
from skimage import io
import cv2
from torch.utils.data import Dataset
from lib.utils.homography import sample_homography, find_map_and_mask, refine_homography
from lib.utils.keypoint import sample_uniform_keypoints
from lib.utils.convert import normalize_torch, gray2RGB


class COCO(Dataset):
    def __init__(self, root, transforms=None, size=(240, 320), num_kps=3000,
                 mode='train'):
        """
        COCO dataset with random homography.
        
        :param root: COCO2017 root
        :param size: (h, w)
        :param num_kps: number of truth keypoints
        :param mode: 'train', 'val' or 'test'
        """
        Dataset.__init__(self)
        self.transforms = transforms
        self.size = size
        self.num_kps = num_kps
        self.mode = mode
        self.img_paths = self.get_img_paths(root, mode)
    
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
        img_path = self.img_paths[index]
        h, w = self.size
        # img: (480, 640), uint8, resize to 'size'
        img = io.imread(img_path)
        img = cv2.resize(img, dsize=(w, h))
        
        # there are some gray images in COCO
        img = gray2RGB(img)
        
        # sample homography for left and right image
        H_left = sample_homography(h, w, direction='left')
        H_right = sample_homography(h, w, direction='right')
        
        # refine homography
        H_left, H_right, H = refine_homography(H_left, H_right, h, w)
        # generate the two images
        img_left = cv2.warpPerspective(img, H_left, dsize=(w, h))
        img_right = cv2.warpPerspective(img, H_right, dsize=(w, h))
        
        # find flow map and mask
        # map: int, mask: boolean
        h0, w0 = img_left.shape[:2]
        h1, w1 = img_right.shape[:2]
        map, mask = find_map_and_mask(h0, w0, h1, w1, H)
        
        # find keypoints
        # kps: int, (N, 2)
        kps0, kps1 = sample_uniform_keypoints(img_left, img_right, H, self.num_kps)
        
        # we should assume that only data augmentation are applied
        if self.transforms:
            img_left = self.transforms(img_left)
            img_right = self.transforms(img_right)
        
        # to tensor
        img_left = torch.from_numpy(img_left).permute(2, 0, 1).float()
        img_right = torch.from_numpy(img_right).permute(2, 0, 1).float()
        
        img_left = img_left / 255.0
        img_right = img_right / 255.0

        # normalize
        img_left = normalize_torch(img_left)
        img_right = normalize_torch(img_right)
        
        # keypoints
        kps0 = torch.from_numpy(kps0).long()
        kps1 = torch.from_numpy(kps1).long()
        map = torch.from_numpy(map).long()
        mask = torch.from_numpy(mask.astype(int)).byte()
        
        data = {
            'img0': img_left,
            'img1': img_right,
        }
        targets = {
            'kps0': kps0,
            'kps1': kps1,
            'map': map,
            'mask': mask,
            'H': H,
            'path': img_path
        }
        
        return data, targets
    
    def __len__(self):
        return len(self.img_paths)
    
    def get_img_paths(self, root, mode):
        """
        Retrieve image paths for all images.
        """
        img_paths = []
        dir = {
            'train': 'train2017/*',
            'val': 'val2017/*',
            'test': 'test2017/*',
        }[mode]
        img_paths += glob.glob(os.path.join(root, dir))
        
        return img_paths


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from lib.utils.convert import tonumpyimg, tonumpy
    from lib.utils.keypoint import draw_match
    
    ds = COCO('data/MSCOCO2017')
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
