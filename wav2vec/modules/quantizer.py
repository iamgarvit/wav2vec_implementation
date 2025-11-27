import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantizer(nn.Module):
    def __init__(self, input_dim, G=2, V=320, temp_init=2.0, temp_min=0.5, temp_decay=0.999995):
        super(Quantizer, self).__init__()
        assert input_dim % G == 0, "Input dimension must be divisible by number of groups G."
        self.G = G  
        self.V = V  
        self.input_dim = input_dim
        self.group_dim = input_dim // G
        self.codebooks = nn.Parameter(torch.randn(G, V, input_dim // G))
        self.proj = nn.Linear(input_dim, G * V)
        self.register_buffer("_temperature", torch.tensor(float(temp_init)), persistent=True)
        self.temp_min = float(temp_min)
        self.temp_decay = float(temp_decay)

    @property
    def temperature(self):
        return float(self._temperature.item())

    def update_temperature(self):
        new_temp = max(self.temp_min, self.temperature * self.temp_decay)
        self._temperature.fill_(new_temp)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("Expected shape: [B, T, C]")
        
        logits = self.proj(x)                              # [B, T, G * V]
        B, T, _ = logits.shape
        logits = logits.view(B, T, self.G, self.V)         # [B, T, G, V]
        probs = F.gumbel_softmax(logits=logits, tau=self.temperature, dim=-1, hard=True)
        cb = self.codebooks.unsqueeze(0).unsqueeze(0)    # [1, 1, G, V, C/G]
        probs_expanded = probs.unsqueeze(-1)                  # [B, T, G, V, 1]
        quantized = torch.sum(probs_expanded * cb, dim=-2)   # [B, T, G, C/G]
        quantized = quantized.view(B, T, self.input_dim)      # [B, T, C]

        soft = torch.softmax(logits, dim=-1)
        p_bar = soft.mean(dim=(0,1))  
        eps = 1e-9
        diversity_loss = (p_bar * torch.log(p_bar + eps)).sum() / (self.G * self.V)
        self.update_temperature()
        return quantized, diversity_loss