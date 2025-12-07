import torch
import torch.nn as nn
from .transformer_layer import TransformerLayer
from .positional_embedding import PositionalEmbedding

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_layers=12, num_heads=8, 
                 ffn_dim=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.pos_embedding = PositionalEmbedding(embed_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, padding_mask=None):
        x = self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)
        
        return x
