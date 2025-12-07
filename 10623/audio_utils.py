# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Audio processing utilities for training and evaluation.
"""

import torch
import torchaudio
from typing import Optional, Tuple
import numpy as np


def load_audio(
    path: str,
    sample_rate: int = 32000,
    mono: bool = True,
    max_duration: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """Load audio file and resample if needed.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        max_duration: Maximum duration in seconds (truncate if longer)
        
    Returns:
        Tuple of (waveform, actual_sample_rate)
        waveform shape: [C, T] or [T] if mono
    """
    waveform, sr = torchaudio.load(path)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Truncate if needed
    if max_duration is not None:
        max_samples = int(max_duration * sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
    
    return waveform, sample_rate


def normalize_audio(waveform: torch.Tensor, method: str = 'peak') -> torch.Tensor:
    """Normalize audio waveform.
    
    Args:
        waveform: Audio waveform
        method: Normalization method ('peak', 'rms', 'lufs')
        
    Returns:
        Normalized waveform
    """
    if method == 'peak':
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
    elif method == 'rms':
        rms = (waveform ** 2).mean().sqrt()
        if rms > 0:
            waveform = waveform / rms
    # TODO: Add LUFS normalization if needed
    
    return waveform


def mix_audio(
    audio1: torch.Tensor,
    audio2: torch.Tensor,
    mix_ratio: float = 0.5,
) -> torch.Tensor:
    """Mix two audio signals.
    
    Args:
        audio1: First audio [C, T] or [T]
        audio2: Second audio [C, T] or [T]
        mix_ratio: Mix ratio (0.0 = only audio1, 1.0 = only audio2)
        
    Returns:
        Mixed audio
    """
    # Ensure same shape
    if audio1.dim() != audio2.dim():
        if audio1.dim() == 1:
            audio1 = audio1.unsqueeze(0)
        if audio2.dim() == 1:
            audio2 = audio2.unsqueeze(0)
    
    # Pad to same length
    max_len = max(audio1.shape[-1], audio2.shape[-1])
    if audio1.shape[-1] < max_len:
        padding = torch.zeros(
            *audio1.shape[:-1], max_len - audio1.shape[-1],
            device=audio1.device, dtype=audio1.dtype
        )
        audio1 = torch.cat([audio1, padding], dim=-1)
    if audio2.shape[-1] < max_len:
        padding = torch.zeros(
            *audio2.shape[:-1], max_len - audio2.shape[-1],
            device=audio2.device, dtype=audio2.dtype
        )
        audio2 = torch.cat([audio2, padding], dim=-1)
    
    # Mix
    mixed = (1 - mix_ratio) * audio1 + mix_ratio * audio2
    
    # Normalize to prevent clipping
    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak
    
    return mixed


def compute_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: Optional[int] = None,
) -> torch.Tensor:
    """Compute spectrogram or mel-spectrogram.
    
    Args:
        waveform: Audio waveform [C, T] or [T]
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bins (if None, returns linear spectrogram)
        
    Returns:
        Spectrogram [F, T] or [C, F, T]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    if n_mels is not None:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,  # Default, adjust if needed
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    else:
        transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
        )
    
    spec = transform(waveform)
    
    # Convert to log scale
    spec = torch.log10(spec + 1e-10)
    
    return spec.squeeze(0) if spec.shape[0] == 1 else spec

