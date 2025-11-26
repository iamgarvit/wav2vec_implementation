import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from wav2vec.modules.encoder import Encoder

if __name__ == "__main__":
    enc = Encoder(seed=0)
    x = torch.randn(5, 16000)
    expected_out_lengths = enc.compute_output_lengths([16000, 8000, 8000, 16000, 12000])
    features, output_lengths, padding = enc(x, input_lengths=[16000, 8000, 8000, 16000, 12000])
    print("Features shape:", features.shape)
    print("Expected output:", expected_out_lengths)
    print("Actual outputs:", output_lengths)
    print("Padding shape:", padding.shape)
    print("Padding per example:", (padding).sum(dim=1))