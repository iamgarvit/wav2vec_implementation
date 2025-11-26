import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from wav2vec.modules.encoder import Encoder

def main():
    ngpu = torch.cuda.device_count()
    print("GPUs:", ngpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder(seed=0)
    if ngpu > 1:
        enc = nn.DataParallel(enc)
    enc = enc.to(device)
    enc.eval()

    B = 8
    L = 16000
    batch = torch.randn(B, L, device=device)
    lengths = [L] * B

    features, output_lengths, mask = enc(batch, input_lengths=lengths)
    print("features.shape:", features.shape)
    print("output_lengths:", output_lengths)
    print("mask.shape:", mask.shape)

if __name__ == "__main__":
    main()
