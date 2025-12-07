import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wav2vec.modules.wav2vec import Wav2Vec
from wav2vec.modules.data_loader import LibriSpeechDataset


def collate_fn(batch):
    waveforms = [item['waveform'].squeeze(0) for item in batch]  
    lengths = torch.tensor([len(waveform) for waveform in waveforms], dtype=torch.long)
    
    max_len = lengths.max().item()
    padded_audios = torch.zeros(len(waveforms), max_len)
    
    for i, waveform in enumerate(waveforms):
        padded_audios[i, :len(waveform)] = waveform
    
    return {
        'audio': padded_audios,
        'lengths': lengths
    }


def train_epoch(model, dataloader, optimizer, device, epoch, num_negatives=100, temperature=0.1, diversity_weight=0.1):
    model.train()
    
    total_loss = 0.0
    total_contrastive = 0.0
    total_diversity = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        audio = batch['audio'].to(device)
        lengths = batch['lengths'].to(device)
        
        outputs = model(audio, lengths)
        
        loss, loss_dict = model.compute_loss(
            outputs,
            num_negatives=num_negatives,
            temperature=temperature,
            diversity_weight=diversity_weight
        )
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss_dict['total_loss']
        total_contrastive += loss_dict['contrastive_loss']
        total_diversity += loss_dict['diversity_loss']
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {loss_dict['total_loss']:.4f} | "
                  f"Contrastive: {loss_dict['contrastive_loss']:.4f} | "
                  f"Diversity: {loss_dict['diversity_loss']:.4f} | "
                  f"Time: {elapsed:.2f}s")
            start_time = time.time()
    
    avg_loss = total_loss / num_batches
    avg_contrastive = total_contrastive / num_batches
    avg_diversity = total_diversity / num_batches
    
    return {
        'loss': avg_loss,
        'contrastive_loss': avg_contrastive,
        'diversity_loss': avg_diversity
    }


def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename=None):
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wav2vec_epoch{epoch}_{timestamp}.pt"
    
    filepath = os.path.join(save_dir, filename)
    
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, filepath)    
    return filepath


def train(
    data_dir,
    save_dir,
    num_epochs=10,
    batch_size=8,
    learning_rate=5e-4,
    num_negatives=100,
    temperature=0.1,
    diversity_weight=0.1,
    save_every=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print("-"*60)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-"*60)
    
    print("\nLoading dataset")
    dataset = LibriSpeechDataset(data_dir)
    print(f"Dataset size: {len(dataset)} samples")
    
    num_workers = 4 if device == 'cuda' else 0
    pin_memory = device == 'cuda'
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print("\nInitializing model")
    model = Wav2Vec(
        in_channels=1,
        conv_channels=[512, 512, 512, 512, 512, 512, 512],
        kernel_sizes=[10, 3, 3, 3, 3, 2, 2],
        strides=[5, 2, 2, 2, 2, 2, 2],
        encoder_dropout=0.1,
        mask_prob=0.065,
        mask_length=10,
        embed_dim=512,
        num_transformer_layers=12,
        num_heads=8,
        ffn_dim=2048,
        transformer_dropout=0.1,
        num_groups=2,
        num_vars=320,
        temp_init=2.0,
        temp_min=0.5,
        temp_decay=0.999995,
        sample_rate=16000,
        seed=42
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    elif device == 'cuda':
        print(f"Using single GPU")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    
    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        print(f"EPOCH {epoch}/{num_epochs}")
        
        metrics = train_epoch(
            model, dataloader, optimizer, device, epoch,
            num_negatives=num_negatives,
            temperature=temperature,
            diversity_weight=diversity_weight
        )
        
        # Print epoch summary
        print(f"Epoch {epoch} Summary:")
        print(f"  Average Loss: {metrics['loss']:.4f}")
        print(f"  Contrastive Loss: {metrics['contrastive_loss']:.4f}")
        print(f"  Diversity Loss: {metrics['diversity_loss']:.4f}")
        
        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, metrics['loss'], save_dir)
        
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_checkpoint(model, optimizer, epoch, metrics['loss'], save_dir, 
                          filename='best_model.pt')
    
    save_checkpoint(model, optimizer, num_epochs, metrics['loss'], save_dir, 
                   filename='final_model.pt')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Wav2Vec model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Training parameters
    NUM_EPOCHS = 2
    BATCH_SIZE = 2
    LEARNING_RATE = 5e-4
    NUM_NEGATIVES = 100
    TEMPERATURE = 0.1
    DIVERSITY_WEIGHT = 0.1
    SAVE_EVERY = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_negatives=NUM_NEGATIVES,
        temperature=TEMPERATURE,
        diversity_weight=DIVERSITY_WEIGHT,
        save_every=SAVE_EVERY,
        device=DEVICE
    )