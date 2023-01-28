import torch
from torch import nn

class Gaussian_Mixture_Mask(nn.Module):
    def __init__(self, num_patches, num_kernals):
        super(Gaussian_Mixture_Mask, self).__init__()
        self.num_kernals = num_kernals
        self.sigma = nn.Parameter(torch.randn(num_kernals, 1, 1) * 10 + 10)
        self.alpha = nn.Parameter(torch.randn(num_kernals, 1, 1) * 2)
        self.mask = nn.Parameter(self.On_attention_gaussian_mask(num_patches, num_kernals), requires_grad=False)

    def forward(self, x):
        attn = torch.sum(self.alpha * self.mask ** (1 / (self.sigma ** 2 + 1e-5)), 0)
        return attn * x

    def On_attention_gaussian_mask(self, num_patches, num_kernals):
        mask = torch.zeros(num_patches, num_patches)
        img_size = num_patches ** 0.5
        for i in range(num_patches):
            for j in range(num_patches):
                x_change = i % img_size - j % img_size
                y_change = i // img_size - j // img_size
                mask[i, j] = - (x_change ** 2 + y_change ** 2)
        mask = torch.exp(mask) ** (1 / 2)
        return mask.expand(num_kernals, num_patches, num_patches)