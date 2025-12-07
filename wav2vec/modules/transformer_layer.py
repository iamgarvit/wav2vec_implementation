import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, embed_dim),
                                 nn.Dropout(dropout))
        
    def forward(self, x, padding_mask=None):
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x, key_padding_mask=padding_mask)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x