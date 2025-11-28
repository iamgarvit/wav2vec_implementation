import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.masker_helper import get_masker_helper

class Masker(nn.Module):
    def __init__(self, embed_dim, mask_prob=0.065, mask_length=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_prob = float(mask_prob)
        self.mask_length = int(mask_length)
        self.mask_emb = nn.Parameter(torch.randn(1, 1, embed_dim))

    def get_mask(self, B, T, device=None):
        if device is None:
            device = torch.device('cpu')
        return get_masker_helper(B, T, self.mask_prob, self.mask_length, device)
    
    def apply_mask(self, Z, mask):
        assert Z.dim() == 3, "Expected Z shape: [B, T, C]"
        B, T, C = Z.shape
        if mask.shape != (B, T):
            raise ValueError(f"Mask shape {mask.shape} does not match Z shape {Z.shape}")

        if not mask.any():
            b_idx = torch.empty(0, dtype=torch.long, device=Z.device)
            t_idx = torch.empty(0, dtype=torch.long, device=Z.device)
            return Z, b_idx, t_idx, mask

        Z_masked = Z.clone()
        Z_masked[mask] = self.mask_emb
        b_idx, t_idx = torch.where(mask)
        return Z_masked, b_idx, t_idx, mask
    
    def forward(self, Z):
        assert Z.dim() == 3, "Expected Z shape: [B, T, C]"
        B, T, C = Z.shape
        device = Z.device

        if not self.training:
            b_idx = torch.empty(0, dtype=torch.long, device=Z.device)
            t_idx = torch.empty(0, dtype=torch.long, device=Z.device)
            mask = torch.zeros(B, T, dtype=torch.bool, device=Z.device)
            return Z, b_idx, t_idx, mask

        mask = self.get_mask(B, T, device)
        return self.apply_mask(Z, mask)