#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation script for MusicGen fine-tuned on ESC-50.
Computes CLAP similarity, FAD, and other metrics.
"""

import argparse
import os
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
from typing import Dict, List, Optional

try:
    from .musicgen_lora_model import create_musicgen_lora
    from .esc50_dataset import create_esc50_dataloader
    from .clap_utils import CLAPEvaluator
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from musicgen_lora_model import create_musicgen_lora
    from esc50_dataset import create_esc50_dataloader
    from clap_utils import CLAPEvaluator


def compute_fad(real_audio: torch.Tensor, gen_audio: torch.Tensor) -> float:
    """Compute Fréchet Audio Distance (FAD).
    
    Note: This is a simplified version. For full FAD, you need to use
    the official implementation with VGGish embeddings.
    
    Args:
        real_audio: Real audio [B, C, T]
        gen_audio: Generated audio [B, C, T]
        
    Returns:
        FAD score (lower is better)
    """
    # Simplified FAD using spectrogram statistics
    # For full FAD, use: https://github.com/google-research/google-research/tree/master/frechet_audio_distance
    
    def compute_stats(audio):
        # Compute mel-spectrogram
        from torchaudio.transforms import MelSpectrogram
        transform = MelSpectrogram(
            sample_rate=32000,
            n_fft=2048,
            hop_length=512,
            n_mels=64,
        )
        spec = transform(audio)
        spec = torch.log10(spec + 1e-10)
        
        # Compute mean and covariance
        spec_flat = spec.view(spec.shape[0], -1)
        mean = spec_flat.mean(dim=0)
        cov = torch.cov(spec_flat.t())
        return mean, cov
    
    real_mean, real_cov = compute_stats(real_audio)
    gen_mean, gen_cov = compute_stats(gen_audio)
    
    # Compute Fréchet distance
    diff = real_mean - gen_mean
    covmean = torch.matrix_sqrt(real_cov @ gen_cov)
    if torch.isnan(covmean).any():
        return float('inf')
    
    fid = (diff @ diff).sum() + torch.trace(real_cov + gen_cov - 2 * covmean)
    return fid.item()


def compute_kl_divergence(
    real_tokens: torch.Tensor,
    gen_tokens: torch.Tensor,
    vocab_size: int = 2048,
) -> float:
    """Compute KL divergence between token distributions.
    
    Args:
        real_tokens: Real audio tokens [B, K, T]
        gen_tokens: Generated audio tokens [B, K, T]
        vocab_size: Vocabulary size
        
    Returns:
        KL divergence
    """
    # Flatten tokens
    real_flat = real_tokens.view(-1).long()
    gen_flat = gen_tokens.view(-1).long()
    
    # Compute distributions
    real_dist = torch.bincount(real_flat, minlength=vocab_size).float()
    real_dist = real_dist / real_dist.sum()
    real_dist = real_dist + 1e-10  # Avoid log(0)
    
    gen_dist = torch.bincount(gen_flat, minlength=vocab_size).float()
    gen_dist = gen_dist / gen_dist.sum()
    gen_dist = gen_dist + 1e-10
    
    # Compute KL divergence
    kl = (real_dist * torch.log(real_dist / gen_dist)).sum()
    return kl.item()


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    config: dict,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate the model.
    
    Args:
        model: MusicGenLoRA model
        dataloader: Evaluation dataloader
        device: Device to evaluate on
        config: Configuration
        output_dir: Directory to save generated samples
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Initialize CLAP evaluator
    clap_evaluator = None
    try:
        clap_evaluator = CLAPEvaluator(device=device)
    except Exception as e:
        print(f"Warning: Could not initialize CLAP evaluator: {e}")
    
    all_clap_similarities = []
    all_fads = []
    all_kls = []
    all_texts = []
    all_gen_audio = []
    all_real_audio = []
    
    num_samples = config.get('eval_num_samples', 100)
    gen_duration = config.get('gen_duration', 10.0)
    
    print(f"Evaluating on {num_samples} samples...")
    
    for batch_idx, (audio, infos) in enumerate(tqdm(dataloader, desc="Evaluation")):
        if batch_idx * dataloader.batch_size >= num_samples:
            break
        
        audio = audio.to(device)  # [B, C, T]
        batch_size = audio.shape[0]
        
        # Get text descriptions
        texts = []
        for info in infos:
            description = getattr(info, 'description', '')
            if not description:
                description = 'environmental sounds'
            texts.append(description)
        all_texts.extend(texts)
        
        # Generate audio
        attributes = []
        for info in infos:
            from audiocraft.modules.conditioners import ConditioningAttributes
            description = getattr(info, 'description', '')
            attrs = ConditioningAttributes(text={'description': description})
            attributes.append(attrs)
        
        # Generate tokens
        total_gen_len = int(gen_duration * model.compression_model.frame_rate)
        gen_tokens = model.lm.generate(
            None, attributes, max_gen_len=total_gen_len,
            use_sampling=True, temp=1.0, top_k=250, top_p=0.0,
        )
        
        # Decode to audio
        gen_audio = model.compression_model.decode(gen_tokens, None)
        
        # Store for batch evaluation
        all_gen_audio.append(gen_audio.cpu())
        all_real_audio.append(audio.cpu())
        
        # Compute CLAP similarity
        if clap_evaluator:
            try:
                results = clap_evaluator.evaluate_batch(
                    texts, gen_audio, sample_rate=32000
                )
                all_clap_similarities.append(results['mean_similarity'])
            except Exception as e:
                print(f"Warning: CLAP evaluation failed: {e}")
        
        # Compute FAD (on subset to save memory)
        if batch_idx < 5:  # Only compute FAD for first few batches
            try:
                fad = compute_fad(audio, gen_audio)
                if not np.isinf(fad):
                    all_fads.append(fad)
            except Exception as e:
                print(f"Warning: FAD computation failed: {e}")
        
        # Compute KL divergence on tokens
        try:
            real_tokens, _ = model.compression_model.encode(audio)
            kl = compute_kl_divergence(real_tokens, gen_tokens)
            all_kls.append(kl)
        except Exception as e:
            print(f"Warning: KL divergence computation failed: {e}")
        
        # Save samples
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for i in range(batch_size):
                idx = batch_idx * dataloader.batch_size + i
                if idx < num_samples:
                    # Save generated audio
                    gen_path = output_dir / f"gen_{idx:04d}.wav"
                    real_path = output_dir / f"real_{idx:04d}.wav"
                    
                    import torchaudio
                    torchaudio.save(
                        str(gen_path),
                        gen_audio[i].cpu(),
                        sample_rate=32000,
                    )
                    torchaudio.save(
                        str(real_path),
                        audio[i].cpu(),
                        sample_rate=32000,
                    )
                    
                    # Save text description
                    text_path = output_dir / f"text_{idx:04d}.txt"
                    with open(text_path, 'w') as f:
                        f.write(texts[i])
    
    # Aggregate metrics
    metrics = {}
    
    if all_clap_similarities:
        metrics['clap_similarity'] = np.mean(all_clap_similarities)
        metrics['clap_std'] = np.std(all_clap_similarities)
    
    if all_fads:
        metrics['fad'] = np.mean(all_fads)
    
    if all_kls:
        metrics['kl_divergence'] = np.mean(all_kls)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate MusicGen on ESC-50')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to LoRA checkpoint')
    parser.add_argument('--config', type=str, default='config_esc50_lora.yaml',
                        help='Path to config file')
    parser.add_argument('--esc50_root', type=str, default=None,
                        help='Path to ESC-50 dataset root')
    parser.add_argument('--output_dir', type=str, default='./eval_outputs',
                        help='Output directory for generated samples')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate on (train/valid/test)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config
    if args.esc50_root:
        config['dataset']['root'] = args.esc50_root
    if args.device:
        config['device'] = args.device
    
    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading MusicGen model with LoRA...")
    model = create_musicgen_lora(
        model_name=config.get('model_name', 'facebook/musicgen-small'),
        device=device,
        lora_rank=config.get('lora_rank', 8),
        lora_alpha=config.get('lora_alpha', 16.0),
        lora_dropout=config.get('lora_dropout', 0.0),
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.checkpoint}...")
    model.load_lora_weights(args.checkpoint)
    
    # Check ESC50_ROOT
    esc50_root = config['dataset'].get('root') or os.getenv('ESC50_ROOT')
    if not esc50_root:
        raise ValueError(
            "ESC-50 dataset root not specified. "
            "Set ESC50_ROOT environment variable or set dataset.root in config."
        )
    
    # Create dataloader
    print("Creating evaluation dataloader...")
    eval_loader = create_esc50_dataloader(
        root=esc50_root,
        split=args.split,
        batch_size=config['eval'].get('batch_size', 4),
        num_workers=config['eval'].get('num_workers', 4),
        segment_duration=config['dataset'].get('segment_duration', None),
        sample_rate=config['dataset'].get('sample_rate', 32000),
    )
    
    # Evaluate
    output_dir = Path(args.output_dir)
    metrics = evaluate(model, eval_loader, device, config, output_dir)
    
    # Print results
    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()

