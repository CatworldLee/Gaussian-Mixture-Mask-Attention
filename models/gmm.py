import torch
from torch import nn

class Simple_Mask(nn.Module):
    def __init__(self, num_heads, num_patches):
        super(Simple_Mask, self).__init__()
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.mask = nn.Parameter(torch.zeros(num_patches, num_patches))
    
    def forward(self, x):
        return self.mask * x

class Gaussian_Mixture_Mask(nn.Module):
    def __init__(self, num_heads, num_kernals, mask):
        super(Gaussian_Mixture_Mask, self).__init__()
        self.num_kernals = num_kernals
        self.sigma = nn.Parameter(torch.randn(num_kernals, num_heads, 1, 1) * 10 + 10)
        self.alpha = nn.Parameter(torch.randn(num_kernals, num_heads, 1, 1) * 2)
        self.mask = mask

    def forward(self, x):
        attn = torch.sum(self.alpha * self.mask ** (1 / (self.sigma ** 2 + 1e-5)), 0)
        return attn * x
    
def On_attention_gaussian_mask(num_patches):
    mask = torch.zeros(num_patches, num_patches)
    img_size = num_patches ** 0.5
    for i in range(num_patches):
        for j in range(num_patches):
            x_change = i % img_size - j % img_size
            y_change = i // img_size - j // img_size
            mask[i, j] = - (x_change ** 2 + y_change ** 2)
    mask = torch.exp(mask) ** (1 / 2)
    return mask.expand(1, 1, num_patches, num_patches)

