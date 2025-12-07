import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
    
    def forward(self, x):
        B, T, C = x.shape
        return x + self.pos_emb[:, :T, :]