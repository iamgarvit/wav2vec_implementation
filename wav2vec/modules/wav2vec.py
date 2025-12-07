import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.encoder import Encoder
from modules.masker import Masker
from modules.transformer import TransformerEncoder
from modules.quantizer import Quantizer

class Wav2Vec(nn.Module):
    def __init__(self,
                 # Encoder parameters
                 in_channels=1,
                 conv_channels=[512, 512, 512, 512, 512, 512, 512],
                 kernel_sizes=[10, 3, 3, 3, 3, 2, 2],
                 strides=[5, 2, 2, 2, 2, 2, 2],
                 encoder_dropout=0.1,
                 # Masker parameters
                 mask_prob=0.065,
                 mask_length=10,
                 # Transformer parameters
                 embed_dim=512,
                 num_transformer_layers=12,
                 num_heads=8,
                 ffn_dim=2048,
                 transformer_dropout=0.1,
                 # Quantizer parameters
                 num_groups=2,
                 num_vars=320,
                 temp_init=2.0,
                 temp_min=0.5,
                 temp_decay=0.999995,
                 # Other
                 sample_rate=16000,
                 seed=None):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Encoder
        self.encoder = Encoder(
            seed=seed,
            in_channels=in_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            dropout=encoder_dropout,
            sample_rate=sample_rate
        )
        
        # Masker
        self.masker = Masker(
            embed_dim=embed_dim,
            mask_prob=mask_prob,
            mask_length=mask_length
        )
        
        # Transformer
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=transformer_dropout,
            max_len=5000
        )
        
        # Quantizer
        self.quantizer = Quantizer(
            input_dim=embed_dim,
            G=num_groups,
            V=num_vars,
            temp_init=temp_init,
            temp_min=temp_min,
            temp_decay=temp_decay
        )
        
        # Projection layer 
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, audio, input_lengths):
        # Encoder
        features, output_lengths, padding_mask = self.encoder(audio, input_lengths)
        
        # Masker
        masked_features, mask_indices_b, mask_indices_t, mask = self.masker(features)
        
        # Transformer
        contextualized = self.transformer(masked_features, padding_mask)
        
        # Quantizer
        quantized, diversity_loss = self.quantizer(features)
        
        return {
            'contextualized': contextualized,
            'quantized': quantized,
            'mask_indices_b': mask_indices_b,
            'mask_indices_t': mask_indices_t,
            'diversity_loss': diversity_loss,
            'features': features,
            'padding_mask': padding_mask,
            'output_lengths': output_lengths
        }
    
    def compute_contrastive_loss(self, outputs, num_negatives=100, temperature=0.1):
        contextualized = outputs['contextualized']
        quantized = outputs['quantized']
        mask_indices_b = outputs['mask_indices_b']
        mask_indices_t = outputs['mask_indices_t']
        
        if len(mask_indices_b) == 0:
            return torch.tensor(0.0, device=contextualized.device)
        
        c_masked = contextualized[mask_indices_b, mask_indices_t]  
        q_masked = quantized[mask_indices_b, mask_indices_t]       
        c_masked = self.projection(c_masked)  
        
        # Positive logits
        pos_logits = F.cosine_similarity(c_masked, q_masked, dim=-1)  
        pos_logits = pos_logits / temperature
        
        # Negative sampling
        B, T, C = quantized.shape
        M = c_masked.shape[0]
        
        neg_indices = torch.randint(0, B * T, (M, num_negatives), device=quantized.device)
        quantized_flat = quantized.view(-1, C)
        negatives = quantized_flat[neg_indices] 
        
        c_expanded = c_masked.unsqueeze(1)  
        neg_logits = F.cosine_similarity(c_expanded, negatives, dim=-1)  
        neg_logits = neg_logits / temperature
        
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  
        labels = torch.zeros(M, dtype=torch.long, device=logits.device)  
        
        contrastive_loss = F.cross_entropy(logits, labels)
        
        return contrastive_loss
    
    def compute_loss(self, outputs, num_negatives=100, temperature=0.1, diversity_weight=0.1):
        contrastive_loss = self.compute_contrastive_loss(outputs, num_negatives, temperature)
        diversity_loss = outputs['diversity_loss']
        
        total_loss = contrastive_loss + diversity_weight * diversity_loss
        
        loss_dict = {
            'total_loss': total_loss.mean().item() if total_loss.numel() > 1 else total_loss.item(),
            'contrastive_loss': contrastive_loss.mean().item() if contrastive_loss.numel() > 1 else contrastive_loss.item(),
            'diversity_loss': diversity_loss.mean().item() if diversity_loss.numel() > 1 else diversity_loss.item()
        }
        
        return total_loss, loss_dict