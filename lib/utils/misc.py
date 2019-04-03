import torch.nn.functional as F
import torch


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


