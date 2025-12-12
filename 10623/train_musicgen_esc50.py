"""
Training script for fine-tuning MusicGen-small on ESC-50 using LoRA.
"""

import argparse
import os
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import json
from typing import Dict, List, Optional

try:
    from .musicgen_lora_model import create_musicgen_lora
    from .esc50_dataset import create_esc50_dataloader
    from .lora import get_lora_parameters
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from musicgen_lora_model import create_musicgen_lora
    from esc50_dataset import create_esc50_dataloader
    from lora import get_lora_parameters


def compute_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy loss per codebook.
    
    Based on audiocraft/solvers/musicgen.py
    """
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    
    ce = torch.zeros([], device=targets.device, dtype=logits.dtype)
    ce_per_codebook = []
    
    for k in range(K):
        logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
        targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
        mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
        
        # Only compute loss on valid tokens
        ce_targets = targets_k[mask_k]
        ce_logits = logits_k[mask_k]
        
        if ce_targets.numel() > 0:
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        else:
            ce_per_codebook.append(torch.tensor(0.0, device=targets.device))
    
    # Average across codebooks
    if K > 0:
        ce = ce / K
    
    return ce


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: dict,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (audio, infos) in enumerate(pbar):
        audio = audio.to(device)
        
        # Encode to tokens
        with torch.no_grad():
            audio_tokens, scale = model.compression_model.encode(audio)
            assert scale is None
        
        # Prepare text conditioning
        attributes = []
        for info in infos:
            from audiocraft.modules.conditioners import ConditioningAttributes
            description = getattr(info, 'description', '')
            if not description:
                description = 'environmental sounds'
            attrs = ConditioningAttributes(text={'description': description})
            attributes.append(attrs)
        
        # Apply dropout for CFG
        attributes = model.lm.cfg_dropout(attributes)
        attributes = model.lm.att_dropout(attributes)
        tokenized = model.lm.condition_provider.tokenize(attributes)
        
        # Get condition tensors
        use_amp = config.get('use_amp', True) and device == 'cuda' and torch.cuda.is_available()
        if use_amp:
            with torch.cuda.amp.autocast():
                condition_tensors = model.lm.condition_provider(tokenized)
        else:
            condition_tensors = model.lm.condition_provider(tokenized)
        
        # Padding mask
        B, K, T = audio_tokens.shape
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=device)
        
        # Forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                model_output = model.lm.compute_predictions(
                    audio_tokens, [], condition_tensors
                )
                logits = model_output.logits
                mask = padding_mask & model_output.mask
        else:
            model_output = model.lm.compute_predictions(
                audio_tokens, [], condition_tensors
            )
            logits = model_output.logits
            mask = padding_mask & model_output.mask
        
        # Compute loss
        loss = compute_cross_entropy(logits, audio_tokens, mask)
        
        # Skip if loss is NaN/Inf
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss: {loss.item()}, skipping batch")
            continue
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check for NaN gradients
        has_nan_grad = False
        for param in get_lora_parameters(model):
            if param.grad is not None and not torch.isfinite(param.grad).all():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"Warning: NaN gradients, skipping batch")
            optimizer.zero_grad()
            continue
        
        # Gradient clipping
        if config.get('max_grad_norm', 0) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                get_lora_parameters(model), config['max_grad_norm']
            )
            if not torch.isfinite(grad_norm):
                print(f"Warning: Non-finite grad norm, skipping batch")
                optimizer.zero_grad()
                continue
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_ce += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
        })
    
    metrics = {
        'loss': total_loss / num_batches,
        'ce': total_ce / num_batches,
    }
    
    return metrics


def validate(
    model,
    dataloader: DataLoader,
    device: str,
    config: dict,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for audio, infos in tqdm(dataloader, desc="Validation"):
            audio = audio.to(device)
            
            # Encode to tokens
            audio_tokens, scale = model.compression_model.encode(audio)
            assert scale is None
            
            # Prepare text conditioning
            attributes = []
            for info in infos:
                from audiocraft.modules.conditioners import ConditioningAttributes
                description = getattr(info, 'description', '')
                if not description:
                    description = 'environmental sounds'
                attrs = ConditioningAttributes(text={'description': description})
                attributes.append(attrs)
            
            tokenized = model.lm.condition_provider.tokenize(attributes)
            
            use_amp = config.get('use_amp', True) and device == 'cuda' and torch.cuda.is_available()
            if use_amp:
                with torch.cuda.amp.autocast():
                    condition_tensors = model.lm.condition_provider(tokenized)
            else:
                condition_tensors = model.lm.condition_provider(tokenized)
            
            # Compute predictions
            if use_amp:
                with torch.cuda.amp.autocast():
                    model_output = model.lm.compute_predictions(
                        audio_tokens, [], condition_tensors
                    )
                    logits = model_output.logits
                    B, K, T = audio_tokens.shape
                    padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=device)
                    mask = padding_mask & model_output.mask
            else:
                model_output = model.lm.compute_predictions(
                    audio_tokens, [], condition_tensors
                )
                logits = model_output.logits
                B, K, T = audio_tokens.shape
                padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=device)
                mask = padding_mask & model_output.mask
            
            # Compute loss
            loss = compute_cross_entropy(logits, audio_tokens, mask)
            
            total_loss += loss.item()
            total_ce += loss.item()
            num_batches += 1
    
    metrics = {
        'loss': total_loss / num_batches,
        'ce': total_ce / num_batches,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train MusicGen on ESC-50 with LoRA')
    parser.add_argument('--config', type=str, default='config_esc50_lora.yaml',
                        help='Path to config file')
    parser.add_argument('--esc50_root', type=str, default=None,
                        help='Path to ESC-50 dataset root (overrides config)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.esc50_root:
        config['dataset']['root'] = args.esc50_root
    if args.device:
        config['device'] = args.device
    
    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading MusicGen model with LoRA...")
    model = create_musicgen_lora(
        model_name=config.get('model_name', 'facebook/musicgen-small'),
        device=device,
        lora_rank=config.get('lora_rank', 8),
        lora_alpha=config.get('lora_alpha', 16.0),
        lora_dropout=config.get('lora_dropout', 0.0),
    )
    
    # Check ESC50_ROOT
    esc50_root = config['dataset'].get('root') or os.getenv('ESC50_ROOT')
    if not esc50_root:
        raise ValueError(
            "ESC-50 dataset root not specified. "
            "Set ESC50_ROOT environment variable or set dataset.root in config."
        )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_esc50_dataloader(
        root=esc50_root,
        split='train',
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        segment_duration=config['dataset'].get('segment_duration', None),
        sample_rate=config['dataset'].get('sample_rate', 32000),
    )
    
    val_loader = create_esc50_dataloader(
        root=esc50_root,
        split='valid',
        batch_size=config['training'].get('val_batch_size', config['training']['batch_size']),
        num_workers=config['training'].get('num_workers', 4),
        segment_duration=config['dataset'].get('segment_duration', None),
        sample_rate=config['dataset'].get('sample_rate', 32000),
    )
    
    # Create optimizer (only for LoRA parameters)
    lora_params = get_lora_parameters(model)
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01),
    )
    
    # Learning rate scheduler
    scheduler = None
    if config['training'].get('scheduler', '') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_lora_weights(args.resume.replace('.pt', '_lora.pt'))
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['training']['epochs']} ===")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, config)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'optimizer': optimizer.state_dict(),
        }
        if scheduler:
            checkpoint['scheduler'] = scheduler.state_dict()
        
        # Save model checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save LoRA weights separately
        lora_path = output_dir / f'checkpoint_epoch_{epoch + 1}_lora.pt'
        model.save_lora_weights(str(lora_path))
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = output_dir / 'best_checkpoint.pt'
            best_lora_path = output_dir / 'best_checkpoint_lora.pt'
            torch.save(checkpoint, best_path)
            model.save_lora_weights(str(best_lora_path))
            print(f"Saved best model (val_loss={best_val_loss:.4f})")
    
    print("Training complete")


if __name__ == '__main__':
    main()

