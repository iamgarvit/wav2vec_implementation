import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, 
                 seed=None, 
                 in_channels=1, 
                 conv_channels=[512, 512, 512, 512, 512, 512, 512],
                 kernel_sizes=[10, 3, 3, 3, 3, 2, 2],
                 strides=[5, 2, 2, 2, 2, 2, 2],
                 dropout=0.1,
                 sample_rate=16000):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.in_channels = in_channels
        self.conv_channels = list(conv_channels)
        self.kernel_sizes = list(kernel_sizes)
        self.strides = list(strides)
        self.dropout_prob = dropout
        self.sample_rate = sample_rate

        assert(len(self.conv_channels) == len(self.kernel_sizes) == len(self.strides))
        self.paddings = [k // 2 for k in self.kernel_sizes]

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()
        prev_channels = self.in_channels

        for out_c, k, s, p in zip(self.conv_channels, self.kernel_sizes, self.strides, self.paddings):
            conv = nn.Conv1d(in_channels=prev_channels, out_channels=out_c, kernel_size=k, stride=s, padding=p, bias=False)
            self.conv_layers.append(conv)
            self.norm_layers.append(nn.LayerNorm(out_c))
            self.activations.append(nn.GELU())
            self.dropouts.append(nn.Dropout(p=self.dropout_prob))
            prev_channels = out_c
        self._init_weights()    

    def _init_weights(self):
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')

        for ln in self.norm_layers:
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

    def compute_output_lengths(self, input_lengths):
        if isinstance(input_lengths, int):
            L = torch.tensor([input_lengths], dtype=torch.long)
        elif isinstance(input_lengths, (list, tuple)):
            L = torch.tensor(list(input_lengths), dtype=torch.long)
        elif torch.is_tensor(input_lengths):
            L = input_lengths.to(torch.long).clone()
            if L.dim() == 0:
                L = L.unsqueeze(0)
        else:
            raise ValueError("input_lengths must be an int, list, tuple, or torch.Tensor")
        
        for k, s, p in zip(self.kernel_sizes, self.strides, self.paddings):
            L = (L + 2 * int(p) - int(k)) // int(s) + 1
            L = torch.clamp(L, min=1)
        return L.long()
    
    def forward(self, x, input_lengths=None):
        """
        input_lengths are in number of samples
        output is in frames
        padding mask returns true for padded positions
        """
        if (x.dim() == 2):
            x = x.unsqueeze(1)
        elif (x.dim() == 3):
            pass
        else:
            raise ValueError("Input tensor must be (B, T) or (B, 1, T)")
        
        for i, (conv, norm, activ, drop) in enumerate(zip(self.conv_layers, self.norm_layers, self.activations, self.dropouts)):
            x = conv(x)                          # (B, C_out, T_out)
            x = x.transpose(1,2)                 # (B, T_out, C_out)
            x = norm(x)                          
            x = activ(x)
            x = drop(x)
            x = x.transpose(1, 2)                # (B, C_out, T_out)

        features = x.transpose(1, 2).contiguous()        # (B, T_encoder, C)

        if input_lengths is None:
            return features
        
        output_lengths = self.compute_output_lengths(input_lengths)
        output_lengths = output_lengths.to(features.device)

        B, T_e, _ = features.shape
        if (output_lengths > T_e).any():
            output_lengths = torch.clamp(output_lengths, max=T_e)

        pos = torch.arange(T_e, device=features.device).unsqueeze(0).expand(B, T_e)
        padded_mask = pos >= output_lengths.unsqueeze(1)

        assert padded_mask.shape == (B, T_e)
        assert output_lengths.dtype == torch.long

        return features, output_lengths, padded_mask