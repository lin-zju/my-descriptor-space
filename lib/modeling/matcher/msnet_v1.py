from torch import nn
import torch
import torch.nn.functional as F

from lib.modeling.backbone.resnet import MultiResNet, resnet18
from lib.modeling.evaluator.constrastive import ConstrastiveEvaluator
from lib.utils.vis_logger import logger
# from lib.utils.visualize import draw_paired_img_desc_torch


class DescExtractor(nn.Module):
    def __init__(self, dim=256):
        super(DescExtractor, self).__init__()
        self.backbone = MultiResNet(resnet18(True), 3)

        self.regress = nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, feats):
        """
        Descriptor from feature
        :param feats: (B, D, H, W)
        """
        feats = self.backbone(feats)['C3']
        _, _, h, w = feats.size()

        descs = F.normalize(self.regress(feats), p=2, dim=1)
        
        feats_down = F.interpolate(feats, scale_factor=0.5, mode='bilinear')
        down_descs = F.interpolate(self.regress(feats_down), (h, w), mode='bilinear', align_corners=False)

        # (B, 2, D, H ,W)
        descs = torch.stack([descs, down_descs], dim=1)


        return descs



class MSNetV1(nn.Module):
    """
    Simple descritpor network without any feature. The features are extracted
    directly from C3 layer
    """
    def __init__(self):
        nn.Module.__init__(self)

        # C3 size: 128
        self.desc_extractor = DescExtractor(dim=128)
        self.desc_evaluator = ConstrastiveEvaluator()

    def forward(self, data, targets=None):
        """
        :param data:
            img0: (B, 3, H, W)
            img1: (B, 3, H, W)
        :param targets:
            kps0: (B, N, 2)
            kps1: (B, N, 2)
            map: (B, N, H, W, 2)
            mask: (B, N, H, W)
            H: (B, 3, 3)
        """
        images0, images1 = data['img0'], data['img1']
        
        # compute descriptor from C3
        descs0 = self.desc_extractor(images0)
        descs1 = self.desc_extractor(images1)
        
        if targets is None:
            return dict(
                descs0=descs0,
                descs1=descs1
            )
        
        # note, only the original sized descriptor is used for hard mining
        loss, distance, similarity = self.desc_evaluator(
            descs0[:, 0], targets['kps0'], images0,
            descs1[:, 0], targets['kps1'], images1,
            thresh=4, interval=4
        )
        
        loss_dict = dict(loss=loss, distance=distance, similarity=similarity)

        # keep descriptors for visualization
        logger.update(image0=images0[0], image1=images1[0])
        logger.update(kps0=targets['kps0'][0], kps1=targets['kps1'][0])
        logger.update(descs0=descs0[0][0], descs1=descs1[0][0])
        # logger.update(**loss_dict)
        # logger.update(H=targets['H'][0])

        # return dict(loss=loss)
        return loss_dict

    # def inference(self, data):
    #     """
    #     :param images: (B, 3, H, W)
    #     """
    #     img0, img1 = data['']
    #     _, _, h, w = images.size()[-2:]
    #     descs = self.desc_extractor(images)
    #     # interpolate and normalize
    #     descs = F.normalize(F.interpolate(descs, (w, h)), p=2, dim=1)
    #
    #     return descs

