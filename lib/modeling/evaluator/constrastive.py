import torch

from lib.utils.misc import sample_descriptor
from lib.utils.hard_mining.hard_example_mining_layer import hard_negative_mining
from lib.utils.vis_logger import logger


class ConstrastiveEvaluator(object):
    def __call__(self, descs0, kps0, imgs0, descs1, kps1, imgs1, thresh=4, interval=4):
        """
        Compute constrastive loss with hard negative mining
        
        :param descs0 descs1: (B, D, H', W'), downsampled
        :param kps0, kps1: (B, N, 2), original image scale
        :param imgs0, imgs1: (B, 3, H, W)
        :param thresh: mining threshold. NOTE, this is measure at descritpor map scale
        :param interval: mining interval. NOTE, this is measure at descritpor map scale
        :return:
            loss: total loss
            distance: distance between true correspondences
            similarity: similarity between false correspondences
        """
        descs0 = sample_descriptor(descs0, kps0, imgs0)  # [B, N, D]
        # descs2 = sample_descriptor(descr_maps1, kps2)
        descs2, kps2 = hard_negative_mining(descs0, descs1, kps1, imgs1, thresh, interval)  # [B, N, D]
        logger.update(kps2=kps2[0])
        descs1 = sample_descriptor(descs1, kps1, imgs1)  # [B, N, D]

        pos_dist = torch.norm(descs0 - descs1, 2, dim=2)
        neg_dist = torch.norm(descs0 - descs2, 2, dim=2)

        distance = torch.sum(pos_dist) / pos_dist.numel()
        # print(distance)

        similarity = 0.5 - neg_dist
        weight = similarity > 0
        similarity = torch.sum(similarity[weight]) / torch.clamp(torch.sum(weight).float(), min=1.)

        loss = distance + similarity

        return loss, distance, similarity
