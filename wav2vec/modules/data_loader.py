import os
import pathlib
import sys
import glob
import pandas as pd
from torch.utils.data import Dataset
import torch
import soundfile as sf
import numpy as np

class LibriSpeechDataset(Dataset):
    def __init__(self, directory_path):
        self.directory = pathlib.Path(directory_path)
        self.samples = []

        for speaker in os.listdir(directory_path):
            speaker_path = os.path.join(directory_path, speaker)
            if not os.path.isdir(speaker_path):
                continue

            for book in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, book)
                if not os.path.isdir(chapter_path):
                    continue

                files = os.listdir(chapter_path)
                transcript_file = glob.glob(os.path.join(chapter_path, "*.trans.txt"))
                if not transcript_file:
                    transcript_file = glob.glob(os.path.join(chapter_path, "*.txt"))
                if not transcript_file:
                    continue

                transcript_file = transcript_file[0]
                transcript_dict = {}
                with open(transcript_file, 'r') as f:
                    for line in f:
                        id, text = line.strip().split(' ', 1)
                        transcript_dict[id] = text

                for file in files:
                    if file.lower().endswith('.flac'):
                        base_name = file.replace('.flac', '')
                        audio_path = os.path.join(chapter_path, file)

                        transcript = transcript_dict.get(base_name)
                        if transcript is None:
                            continue

                        self.samples.append({
                            'id': base_name,
                            'audio_path': audio_path,
                            'transcript': transcript
                        })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data, sr = sf.read(sample['audio_path'], dtype='float32')

        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)
        
        w = waveform.squeeze(0)
        mean = w.mean()
        std = w.std() if w.std() > 1e-6 else 1.0
        w = (w - mean) / std

        waveform = w.unsqueeze(0)
        return {
            'id': sample['id'],
            'waveform': waveform,
            'sr': sr,
            'transcript': sample['transcript'],
            'path': sample['audio_path']
        }
