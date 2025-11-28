import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from modules.quantizer import Quantizer

def run_tests():
    torch.manual_seed(0)

    B, T, C = 4, 5, 16
    G, V = 2, 8

    x = torch.randn(B, T, C)
    quant = Quantizer(input_dim=C, G=G, V=V, temp_init=2.0, temp_min=0.5, temp_decay=0.99)

    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant.to(device)
    x = x.to(device)

    print("Quantizer tests")
    print(f"Input: {x.shape}, input_dim={C}, G={G}, V={V}, group_dim={quant.group_dim}")
    print("Initial temperature:", quant.temperature)

    q, diversity_loss = quant(x)
    
    logits = quant.proj(x)
    logits = logits.view(B, T, G, V)
    probs = torch.softmax(logits, dim=-1)
    p_bar = probs.mean(dim=(0, 1))

    assert q.shape == (B, T, C), f"q shape {q.shape} wrong"
    assert logits.shape == (B, T, G, V)
    assert probs.shape == (B, T, G, V)
    print("Shapes OK")

    # Check probs sum to 1
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "probs do not sum to 1"
    print("probs.sum(dim=-1) all close to 1.0")

    assert p_bar.shape == (G, V)
    print("p_bar row sums:", p_bar.sum(dim=-1).detach().cpu().numpy())

    assert diversity_loss is not None and diversity_loss.dim() == 0
    print("diversity loss:", float(diversity_loss.item()))

    loss = q.sum() + diversity_loss
    loss.backward()

    assert quant.proj.weight.grad is not None, "proj weights have no grad"
    assert quant.codebooks.grad is not None, "codebooks have no grad"
    print("Gradients flow OK")

    print("Temperature after forward:", quant.temperature)
    print("All tests passed.")

if __name__ == "__main__":
    run_tests()
