import torch

def get_masker_helper(B, T, mask_prob=0.065, mask_length=10, device=None):
    if device is None:
        device = torch.device('cpu')
    
    if T <= 0 or B <= 0:
        raise ValueError("T and B must be positive integers.")
    
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask_length = max(1, min(mask_length, T))

    target_num_mask = int(mask_prob * T)
    num_spans = max(1, target_num_mask // mask_length)

    for b in range(B):
        starts = torch.randint(0, T, (num_spans,), device=device)
        
        for start in starts:
            end = min(start + mask_length, T)
            mask[b, start:end] = True
    return mask