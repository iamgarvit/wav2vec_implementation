import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from torch.utils.data import DataLoader
from modules.data_loader import LibriSpeechDataset
from multiprocessing import cpu_count

def collate_fn(x):
    return x

def main():
    # 1. Load dataset
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'dev-clean')
    dataset = LibriSpeechDataset(root)

    print("Total samples:", len(dataset))
    # 2. Use DataLoader
    workers = max(1, min(12, cpu_count() - 1))   
    print(f"Using {workers} workers (CPU count: {cpu_count()})")
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,      
        collate_fn=collate_fn
    )

    # 3. Fetch one batch
    start_time = time.time()
    batch = next(iter(dataloader))
    batch_load_time = time.time() - start_time

    print(f"\nBatch size = {len(batch)}")
    print("Batch is a:", type(batch))
    print(f"⏱️  Batch load time: {batch_load_time:.4f} seconds")

    # 4. Inspect each sample
    for i, sample in enumerate(batch):
        print(f"\n--- Sample {i} ---")
        print("ID:         ", sample["id"])
        print("Transcript: ", sample["transcript"])
        print("Waveform shape:", sample["waveform"].shape)
        print("Mean:", float(sample["waveform"].mean()))
        print("Std :", float(sample["waveform"].std()))

        if abs(float(sample["waveform"].mean())) < 1e-3:
            print("✓ mean ≈ 0")
        else:
            print("✗ mean NOT zero")

        if abs(float(sample["waveform"].std()) - 1.0) < 1e-2:
            print("✓ std ≈ 1")
        else:
            print("✗ std NOT 1")

    print("\nTEST COMPLETE")

if __name__ == "__main__":
    main()
