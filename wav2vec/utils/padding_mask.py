import torch

def create_padding_mask(lengths, max_len):
    batch_size = lengths.size(0)
    positions = torch.arange(max_len, device=lengths.device)
    positions = positions.unsqueeze(0).expand(batch_size, max_len)
    lengths = lengths.unsqueeze(1)
    mask = positions >= lengths
    return mask
