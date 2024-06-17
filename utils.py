import random
import torch
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import torch.nn.functional as F

def add_gaussian_poission_noise(gt_data, a, b, seed = 19):
    random.seed(seed)
    variance = a * gt_data + b**2
    noisy_data = gt_data + np.random.normal(loc=0, scale=np.sqrt(variance))
    return noisy_data


def cal_ssim(ref, img):
    data_range = np.max(ref) - np.min(ref)
    return SSIM(ref, img, data_range=data_range)

def cal_psnr(ref, img):
    data_range = np.max(ref) - np.min(ref)
    return PSNR(ref, img, data_range=data_range)

class PD_sampler():
    def __init__(self,scale_factor,device):
        self.select_idx_ = torch.stack(
            [torch.randperm(scale_factor * scale_factor) for _ in range(1024*1024)],dim = 0
        ).to(device)
        self.scale = scale_factor

    def sample_img(self,img, sample_out_channels):

        img_unshuffle = F.pixel_unshuffle(img, self.scale)
        n, c, h, w = img_unshuffle.shape
        img_unshuffle = img_unshuffle.permute(0, 2, 3, 1).reshape(-1, self.scale * self.scale)
        mask = torch.ones_like(img_unshuffle,device= img.device)
        shuffled_indices = torch.randperm(n * h * w).to(img.device)
        select_idx = torch.index_select(self.select_idx_[:n* h * w], dim=0, index=shuffled_indices)

        subsampled_img = torch.gather(img_unshuffle, dim=1, index=select_idx[:, :sample_out_channels])
        mask.scatter_(1, select_idx[:, :sample_out_channels], 0)
        subsampled_img = subsampled_img.reshape(n, h, w, sample_out_channels).permute(0, 3, 1, 2)
        mask = mask.reshape(n, h, w, c).permute(0, 3, 1, 2)
        mask = F.pixel_shuffle(mask, upscale_factor=self.scale)

        return subsampled_img, mask