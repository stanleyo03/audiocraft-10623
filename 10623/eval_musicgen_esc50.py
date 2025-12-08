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
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple

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


def load_audio_file(file_path: Path, target_sr: int = 32000, target_length: Optional[int] = None) -> torch.Tensor:
    """Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        target_length: Optional target length in samples (will pad or trim)
        
    Returns:
        Audio tensor [1, T] or [1, target_length]
    """
    audio, sr = torchaudio.load(str(file_path))
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Pad or trim to target length if specified
    if target_length is not None:
        current_length = audio.shape[1]
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            audio = F.pad(audio, (0, padding))
        elif current_length > target_length:
            # Trim
            audio = audio[:, :target_length]
    
    return audio


def compute_fad(real_audio: torch.Tensor, gen_audio: torch.Tensor, device: str = 'cpu') -> float:
    """Compute Fréchet Audio Distance (FAD).
    
    Note: This is a simplified version. For full FAD, you need to use
    the official implementation with VGGish embeddings.
    
    Args:
        real_audio: Real audio [B, C, T]
        gen_audio: Generated audio [B, C, T]
        device: Device to run computation on
        
    Returns:
        FAD score (lower is better)
    """
    # Simplified FAD using spectrogram statistics
    # For full FAD, use: https://github.com/google-research/google-research/tree/master/frechet_audio_distance
    
    def compute_stats(audio, device):
        # Compute mel-spectrogram
        from torchaudio.transforms import MelSpectrogram
        transform = MelSpectrogram(
            sample_rate=32000,
            n_fft=2048,
            hop_length=512,
            n_mels=64,
        ).to(device)  # Move transform to device
        spec = transform(audio)
        spec = torch.log10(spec + 1e-10)
        
        # Compute mean and covariance
        # Use reshape instead of view to handle non-contiguous tensors
        spec_flat = spec.reshape(spec.shape[0], -1)
        mean = spec_flat.mean(dim=0)
        cov = torch.cov(spec_flat.t())
        return mean, cov
    
    real_mean, real_cov = compute_stats(real_audio, device)
    gen_mean, gen_cov = compute_stats(gen_audio, device)
    
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


@torch.no_grad()
def evaluate_ground_truth_comparison(
    pretrained_model,
    finetuned_model,
    prompts: List[str],
    ground_truth_dir: Path,
    device: str,
    output_dir: Optional[Path] = None,
    gen_duration: float = 10.0,
) -> Dict[str, float]:
    """Evaluate models against ground truth audio clips.
    
    Args:
        pretrained_model: Pretrained MusicGen model (without LoRA weights)
        finetuned_model: Fine-tuned MusicGen model (with LoRA weights)
        prompts: List of text prompts
        ground_truth_dir: Directory containing ground truth audio files
        device: Device to run on
        output_dir: Optional directory to save generated samples
        gen_duration: Duration of generated audio in seconds
        
    Returns:
        Dictionary of metrics
    """
    # Map prompts to audio file names
    prompt_to_file = {
        "soft piano with gentle rain": "piano_rain.wav",
        "calming synth under forest ambience": "synth_forest.wav",
        "acoustic guitar with ocean waves": "guitar_waves.wav",
        "drums with echoes": "drum_echoes.wav",
        "lo-fi piano mixed with city street noise": "lofi_city.wav",
        "slow violin over crackling fire": "violin_fire.wav",
        "relaxing flute with birds chirping": "flute_bird.wav",
        "clarinet in the wind": "clarinet_wind.wav",
        "electronic synth with thunder": "synth_thunder.wav",
        "windchimes with nighttime crickets": "windchime_cricket.wav",
    }
    
    # Initialize CLAP evaluator
    clap_evaluator = None
    try:
        clap_evaluator = CLAPEvaluator(device=device)
    except Exception as e:
        print(f"Warning: Could not initialize CLAP evaluator: {e}")
    
    # Lists to store results
    all_ground_truth_audio = []
    all_pretrained_audio = []
    all_finetuned_audio = []
    all_prompts = []
    
    # Per-prompt metrics storage
    per_prompt_metrics = []
    
    target_sr = 32000
    max_length = int(gen_duration * target_sr)
    
    print(f"Evaluating {len(prompts)} prompts...")
    print(f"Target audio length: {gen_duration:.1f}s ({max_length} samples)")
    print("Note: Ground truth audios will be trimmed/padded to match this length")
    
    for prompt in tqdm(prompts, desc="Generating and evaluating"):
        # Load ground truth audio
        audio_file = prompt_to_file.get(prompt)
        if not audio_file:
            print(f"Warning: No ground truth file found for prompt: {prompt}")
            continue
        
        gt_path = ground_truth_dir / audio_file
        if not gt_path.exists():
            print(f"Warning: Ground truth file not found: {gt_path}")
            continue
        
        # Load ground truth audio and pad/trim to max_length for consistent batch processing
        gt_audio = load_audio_file(gt_path, target_sr=target_sr, target_length=max_length)
        gt_audio = gt_audio.to(device)
        
        # Generate with pretrained model
        # MusicGen doesn't have eval() method, but submodules are already in eval mode
        # Just ensure they're set correctly
        pretrained_model.lm.eval()
        pretrained_model.compression_model.eval()
        pretrained_model.set_generation_params(duration=gen_duration, use_sampling=True, top_k=250)
        pretrained_wav = pretrained_model.generate([prompt])
        pretrained_audio = pretrained_wav[0].to(device)  # [C, T]
        
        # Ensure same length as ground truth (trim or pad to max_length)
        if pretrained_audio.shape[1] > max_length:
            pretrained_audio = pretrained_audio[:, :max_length]
        elif pretrained_audio.shape[1] < max_length:
            padding = max_length - pretrained_audio.shape[1]
            pretrained_audio = F.pad(pretrained_audio, (0, padding))
        
        # Generate with fine-tuned model
        finetuned_model.eval()  # MusicGenLoRA has eval() method
        finetuned_model.set_generation_params(duration=gen_duration, use_sampling=True, top_k=250)
        finetuned_wav = finetuned_model.generate([prompt])
        finetuned_audio = finetuned_wav[0].to(device)  # [C, T]
        
        # Ensure same length as ground truth (trim or pad to max_length)
        if finetuned_audio.shape[1] > max_length:
            finetuned_audio = finetuned_audio[:, :max_length]
        elif finetuned_audio.shape[1] < max_length:
            padding = max_length - finetuned_audio.shape[1]
            finetuned_audio = F.pad(finetuned_audio, (0, padding))
        
        # Store for batch evaluation
        all_ground_truth_audio.append(gt_audio.cpu())
        all_pretrained_audio.append(pretrained_audio.cpu())
        all_finetuned_audio.append(finetuned_audio.cpu())
        all_prompts.append(prompt)
        
        # Save samples if requested
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            idx = len(all_prompts) - 1
            base_name = audio_file.replace('.wav', '')
            
            torchaudio.save(
                str(output_dir / f"{base_name}_ground_truth.wav"),
                gt_audio.cpu(),
                target_sr,
            )
            torchaudio.save(
                str(output_dir / f"{base_name}_pretrained.wav"),
                pretrained_audio.cpu(),
                target_sr,
            )
            torchaudio.save(
                str(output_dir / f"{base_name}_finetuned.wav"),
                finetuned_audio.cpu(),
                target_sr,
            )
            with open(output_dir / f"{base_name}_prompt.txt", 'w') as f:
                f.write(prompt)
    
    # Stack all audio for batch processing
    gt_audio_batch = torch.stack(all_ground_truth_audio)  # [N, 1, T]
    pretrained_audio_batch = torch.stack(all_pretrained_audio)  # [N, C, T]
    finetuned_audio_batch = torch.stack(all_finetuned_audio)  # [N, C, T]
    
    # Ensure pretrained and finetuned are mono
    if pretrained_audio_batch.shape[1] > 1:
        pretrained_audio_batch = pretrained_audio_batch.mean(dim=1, keepdim=True)
    if finetuned_audio_batch.shape[1] > 1:
        finetuned_audio_batch = finetuned_audio_batch.mean(dim=1, keepdim=True)
    
    metrics = {}
    
    # Initialize per-prompt metrics
    per_prompt_metrics = [{'prompt': prompt, 'prompt_idx': i} for i, prompt in enumerate(all_prompts)]
    
    # Compute CLAP similarities (per-prompt)
    if clap_evaluator:
        print("\nComputing CLAP similarities...")
        try:
            # Compute per-prompt CLAP scores
            for i, prompt in enumerate(all_prompts):
                # Ground truth vs prompt
                gt_clap = clap_evaluator.evaluate_batch(
                    [prompt], gt_audio_batch[i:i+1], sample_rate=target_sr
                )
                per_prompt_metrics[i]['clap_ground_truth'] = gt_clap['mean_similarity']
                
                # Pretrained vs prompt
                pretrained_clap = clap_evaluator.evaluate_batch(
                    [prompt], pretrained_audio_batch[i:i+1], sample_rate=target_sr
                )
                per_prompt_metrics[i]['clap_pretrained'] = pretrained_clap['mean_similarity']
                
                # Fine-tuned vs prompt
                finetuned_clap = clap_evaluator.evaluate_batch(
                    [prompt], finetuned_audio_batch[i:i+1], sample_rate=target_sr
                )
                per_prompt_metrics[i]['clap_finetuned'] = finetuned_clap['mean_similarity']
            
            # Compute summary metrics
            metrics['clap_ground_truth'] = np.mean([m['clap_ground_truth'] for m in per_prompt_metrics])
            metrics['clap_pretrained'] = np.mean([m['clap_pretrained'] for m in per_prompt_metrics])
            metrics['clap_finetuned'] = np.mean([m['clap_finetuned'] for m in per_prompt_metrics])
            
            print(f"  Ground truth CLAP: {metrics['clap_ground_truth']:.4f}")
            print(f"  Pretrained CLAP: {metrics['clap_pretrained']:.4f}")
            print(f"  Fine-tuned CLAP: {metrics['clap_finetuned']:.4f}")
        except Exception as e:
            print(f"Warning: CLAP evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compute FAD scores (per-prompt)
    print("\nComputing FAD scores...")
    pretrained_fads = []
    finetuned_fads = []
    
    for i in range(len(all_prompts)):
        try:
            # Pretrained vs ground truth
            fad_pretrained = compute_fad(
                gt_audio_batch[i:i+1].to(device),
                pretrained_audio_batch[i:i+1].to(device),
                device=device
            )
            if not np.isinf(fad_pretrained):
                pretrained_fads.append(fad_pretrained)
                if i < len(per_prompt_metrics):
                    per_prompt_metrics[i]['fad_pretrained'] = fad_pretrained
            else:
                if i < len(per_prompt_metrics):
                    per_prompt_metrics[i]['fad_pretrained'] = float('inf')
            
            # Fine-tuned vs ground truth
            fad_finetuned = compute_fad(
                gt_audio_batch[i:i+1].to(device),
                finetuned_audio_batch[i:i+1].to(device),
                device=device
            )
            if not np.isinf(fad_finetuned):
                finetuned_fads.append(fad_finetuned)
                if i < len(per_prompt_metrics):
                    per_prompt_metrics[i]['fad_finetuned'] = fad_finetuned
            else:
                if i < len(per_prompt_metrics):
                    per_prompt_metrics[i]['fad_finetuned'] = float('inf')
        except Exception as e:
            print(f"Warning: FAD computation failed for sample {i}: {e}")
            if i < len(per_prompt_metrics):
                per_prompt_metrics[i]['fad_pretrained'] = None
                per_prompt_metrics[i]['fad_finetuned'] = None
    
    if pretrained_fads:
        metrics['fad_pretrained'] = np.mean(pretrained_fads)
        print(f"  Pretrained FAD: {metrics['fad_pretrained']:.4f}")
    if finetuned_fads:
        metrics['fad_finetuned'] = np.mean(finetuned_fads)
        print(f"  Fine-tuned FAD: {metrics['fad_finetuned']:.4f}")
    
    # Compute KL divergence (per-prompt)
    print("\nComputing KL divergence...")
    pretrained_kls = []
    finetuned_kls = []
    
    # Use the compression model from fine-tuned model (same as pretrained)
    compression_model = finetuned_model.compression_model
    
    for i in range(len(all_prompts)):
        try:
            # Encode ground truth
            gt_tokens, _ = compression_model.encode(gt_audio_batch[i:i+1].to(device))
            
            # Encode pretrained generated audio
            pretrained_tokens, _ = compression_model.encode(pretrained_audio_batch[i:i+1].to(device))
            kl_pretrained = compute_kl_divergence(gt_tokens, pretrained_tokens)
            pretrained_kls.append(kl_pretrained)
            if i < len(per_prompt_metrics):
                per_prompt_metrics[i]['kl_pretrained'] = kl_pretrained
            
            # Encode fine-tuned generated audio
            finetuned_tokens, _ = compression_model.encode(finetuned_audio_batch[i:i+1].to(device))
            kl_finetuned = compute_kl_divergence(gt_tokens, finetuned_tokens)
            finetuned_kls.append(kl_finetuned)
            if i < len(per_prompt_metrics):
                per_prompt_metrics[i]['kl_finetuned'] = kl_finetuned
        except Exception as e:
            print(f"Warning: KL divergence computation failed for sample {i}: {e}")
            if i < len(per_prompt_metrics):
                per_prompt_metrics[i]['kl_pretrained'] = None
                per_prompt_metrics[i]['kl_finetuned'] = None
    
    if pretrained_kls:
        metrics['kl_pretrained'] = np.mean(pretrained_kls)
        print(f"  Pretrained KL: {metrics['kl_pretrained']:.4f}")
    if finetuned_kls:
        metrics['kl_finetuned'] = np.mean(finetuned_kls)
        print(f"  Fine-tuned KL: {metrics['kl_finetuned']:.4f}")
    
    # Save per-prompt metrics to CSV
    if output_dir and per_prompt_metrics:
        import csv
        csv_path = output_dir / 'per_prompt_metrics.csv'
        fieldnames = ['prompt_idx', 'prompt', 'clap_ground_truth', 'clap_pretrained', 'clap_finetuned',
                     'fad_pretrained', 'fad_finetuned', 'kl_pretrained', 'kl_finetuned']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in per_prompt_metrics:
                # Ensure all fields are present
                row = {field: m.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"\n✓ Per-prompt metrics saved to {csv_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate MusicGen on ESC-50')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to LoRA checkpoint (required for fine-tuned evaluation)')
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
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                        help='Directory containing ground truth audio files for comparison')
    parser.add_argument('--eval_mode', type=str, default='esc50',
                        choices=['esc50', 'ground_truth'],
                        help='Evaluation mode: esc50 (dataset) or ground_truth (custom audio)')
    
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
    
    # Evaluation mode: ground_truth comparison
    if args.eval_mode == 'ground_truth':
        if not args.ground_truth_dir:
            raise ValueError("--ground_truth_dir is required for ground_truth evaluation mode")
        
        ground_truth_dir = Path(args.ground_truth_dir)
        if not ground_truth_dir.exists():
            raise ValueError(f"Ground truth directory not found: {ground_truth_dir}")
        
        # Load prompts from run_inference.py
        prompts = [
            "soft piano with gentle rain",
            "calming synth under forest ambience",
            "acoustic guitar with ocean waves",
            "drums with echoes",
            "lo-fi piano mixed with city street noise",
            "slow violin over crackling fire",
            "relaxing flute with birds chirping",
            "clarinet in the wind",
            "electronic synth with thunder",
            "windchimes with nighttime crickets"
        ]
        
        # Load pretrained model (without LoRA)
        print("Loading pretrained MusicGen model...")
        from audiocraft.models.musicgen import MusicGen
        pretrained_model = MusicGen.get_pretrained(
            config.get('model_name', 'facebook/musicgen-small'),
            device=device
        )
        
        # Load fine-tuned model (with LoRA)
        print("Loading fine-tuned MusicGen model with LoRA...")
        finetuned_model = create_musicgen_lora(
            model_name=config.get('model_name', 'facebook/musicgen-small'),
            device=device,
            lora_rank=config.get('lora_rank', 8),
            lora_alpha=config.get('lora_alpha', 16.0),
            lora_dropout=config.get('lora_dropout', 0.0),
        )
        
        if args.checkpoint:
            print(f"Loading LoRA weights from {args.checkpoint}...")
            finetuned_model.load_lora_weights(args.checkpoint)
        else:
            print("Warning: No checkpoint provided, using untrained LoRA model")
        
        # Evaluate
        output_dir = Path(args.output_dir)
        gen_duration = config.get('gen_duration', 10.0)
        metrics = evaluate_ground_truth_comparison(
            pretrained_model=pretrained_model,
            finetuned_model=finetuned_model,
            prompts=prompts,
            ground_truth_dir=ground_truth_dir,
            device=device,
            output_dir=output_dir,
            gen_duration=gen_duration,
        )
        
        # Print results
        print("\n" + "="*60)
        print("=== Ground Truth Comparison Results ===")
        print("="*60)
        print("\nCLAP Similarities (higher is better):")
        if 'clap_ground_truth' in metrics:
            print(f"  Ground Truth: {metrics['clap_ground_truth']:.4f}")
        if 'clap_pretrained' in metrics:
            print(f"  Pretrained:   {metrics['clap_pretrained']:.4f}")
        if 'clap_finetuned' in metrics:
            print(f"  Fine-tuned:    {metrics['clap_finetuned']:.4f}")
        
        print("\nFAD Scores vs Ground Truth (lower is better):")
        if 'fad_pretrained' in metrics:
            print(f"  Pretrained: {metrics['fad_pretrained']:.4f}")
        if 'fad_finetuned' in metrics:
            print(f"  Fine-tuned:  {metrics['fad_finetuned']:.4f}")
        
        print("\nKL Divergence vs Ground Truth (lower is better):")
        if 'kl_pretrained' in metrics:
            print(f"  Pretrained: {metrics['kl_pretrained']:.4f}")
        if 'kl_finetuned' in metrics:
            print(f"  Fine-tuned:  {metrics['kl_finetuned']:.4f}")
        
        # Save results
        results_path = output_dir / 'ground_truth_comparison_results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    else:
        # Original ESC-50 evaluation mode
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for ESC-50 evaluation mode")
        
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

