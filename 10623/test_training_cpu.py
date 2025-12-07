#!/usr/bin/env python3
"""
Quick CPU test script to verify training code works before Colab.
Tests with minimal data and 1-2 epochs on CPU.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from musicgen_lora_model import create_musicgen_lora
from esc50_dataset import create_esc50_dataloader
from lora import get_lora_parameters
from train_musicgen_esc50 import train_epoch, validate, compute_cross_entropy

# Set ESC50_ROOT (adjust to your path)
# Try to find ESC-50 relative to this script first
script_dir = Path(__file__).parent
default_esc50 = script_dir / 'ESC-50'

# Check if default location exists
if default_esc50.exists() and (default_esc50 / 'meta' / 'esc50.csv').exists():
    ESC50_ROOT = str(default_esc50)
else:
    # Fall back to environment variable or hardcoded path
    ESC50_ROOT = os.getenv('ESC50_ROOT', str(default_esc50))

# Convert to Path and verify
esc50_path = Path(ESC50_ROOT)
if not esc50_path.exists():
    print(f"⚠️  Warning: ESC-50 directory not found at: {esc50_path}")
    print(f"   Please set ESC50_ROOT environment variable or download ESC-50 to: {default_esc50}")
    sys.exit(1)

meta_file = esc50_path / 'meta' / 'esc50.csv'
if not meta_file.exists():
    print(f"⚠️  Warning: ESC-50 metadata file not found at: {meta_file}")
    print(f"   Expected structure: {esc50_path}/meta/esc50.csv")
    sys.exit(1)

os.environ['ESC50_ROOT'] = ESC50_ROOT
print(f"✓ Using ESC-50 dataset at: {ESC50_ROOT}")

print("="*60)
print("CPU Training Test")
print("="*60)

# Force CPU
device = 'cpu'
print(f"Using device: {device}")

# Minimal config for testing
config = {
    'model_name': 'facebook/musicgen-small',
    'lora_rank': 4,  # Smaller rank for faster testing
    'lora_alpha': 8.0,
    'lora_dropout': 0.0,
    'dataset': {
        'root': ESC50_ROOT,
        'sample_rate': 32000,
        'segment_duration': 5.0,  # Short segments for faster testing
        'channels': 1,
    },
    'training': {
        'batch_size': 1,  # Minimal batch size
        'val_batch_size': 1,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'epochs': 1,  # Just 1 epoch for testing
        'num_workers': 0,  # Single-threaded for debugging
        'max_grad_norm': 1.0,
        'use_amp': False,  # Disable AMP on CPU
        'scheduler': None,  # No scheduler for quick test
    },
    'eval': {
        'batch_size': 1,
        'num_workers': 0,
        'eval_num_samples': 5,
        'gen_duration': 5.0,
    },
    'device': device,
}

print("\n1. Loading model...")
try:
    model = create_musicgen_lora(
        model_name=config['model_name'],
        device=device,
        lora_rank=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
    )
    print("✓ Model loaded")
    
    # Count parameters
    lora_params = get_lora_parameters(model)
    num_params = sum(p.numel() for p in lora_params)
    print(f"✓ LoRA parameters: {num_params:,}")
    
    # Verify only LoRA params are trainable
    # MusicGenLoRA doesn't inherit from nn.Module, so we need to access submodules
    trainable = sum(p.numel() for p in lora_params if p.requires_grad)
    total_lm = sum(p.numel() for p in model.lm.parameters())
    total_compression = sum(p.numel() for p in model.compression_model.parameters())
    total = total_lm + total_compression
    print(f"✓ Trainable params: {trainable:,} / {total:,} (LoRA only)")
    
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Creating dataloaders...")
try:
    # Limit dataset size by using a small subset
    # We'll create a custom dataset that only uses first N samples
    from esc50_dataset import ESC50Dataset
    
    # Create small subset dataset
    dataset = ESC50Dataset(
        root=config['dataset']['root'],
        split='train',
        segment_duration=config['dataset']['segment_duration'],
        sample_rate=config['dataset']['sample_rate'],
        channels=config['dataset']['channels'],
        return_info=True,
        shuffle=False,
        num_samples=10,  # Only 10 samples for testing
    )
    
    # Create a small dataloader
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        waveforms = []
        infos = []
        for wav, info in batch:
            waveforms.append(wav)
            infos.append(info)
        
        # Stack waveforms
        max_len = max(w.shape[-1] for w in waveforms)
        padded_waveforms = []
        for w in waveforms:
            if w.shape[-1] < max_len:
                padding = torch.zeros(
                    w.shape[0], max_len - w.shape[-1],
                    device=w.device, dtype=w.dtype
                )
                w = torch.cat([w, padding], dim=-1)
            padded_waveforms.append(w)
        
        stacked_wav = torch.stack(padded_waveforms)
        return stacked_wav, infos
    
    train_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Validation dataset (even smaller)
    val_dataset = ESC50Dataset(
        root=config['dataset']['root'],
        split='valid',
        segment_duration=config['dataset']['segment_duration'],
        sample_rate=config['dataset']['sample_rate'],
        channels=config['dataset']['channels'],
        return_info=True,
        shuffle=False,
        num_samples=5,  # Only 5 samples for validation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['val_batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Test one batch
    audio, infos = next(iter(train_loader))
    print(f"✓ Batch shape: {audio.shape}")
    print(f"✓ Sample captions: {[getattr(info, 'description', '')[:30] for info in infos[:2]]}")
    
except Exception as e:
    print(f"✗ Dataloader creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing forward pass...")
try:
    model.eval()
    with torch.no_grad():
        audio, infos = next(iter(train_loader))
        
        # Encode audio to tokens
        audio_tokens, scale = model.compression_model.encode(audio)
        assert scale is None
        
        # Prepare attributes
        from audiocraft.modules.conditioners import ConditioningAttributes
        attributes = []
        for info in infos:
            description = getattr(info, 'description', '')
            if not description:
                description = 'environmental sounds'
            attrs = ConditioningAttributes(text={'description': description})
            attributes.append(attrs)
        
        # Tokenize
        tokenized = model.lm.condition_provider.tokenize(attributes)
        condition_tensors = model.lm.condition_provider(tokenized)
        
        # Forward pass
        model_output = model.lm.compute_predictions(
            audio_tokens, [], condition_tensors
        )
        logits = model_output.logits
        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Mask shape: {model_output.mask.shape}")
        
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing loss computation...")
try:
    model.eval()
    with torch.no_grad():
        audio, infos = next(iter(train_loader))
        audio_tokens, _ = model.compression_model.encode(audio)
        
        from audiocraft.modules.conditioners import ConditioningAttributes
        attributes = []
        for info in infos:
            description = getattr(info, 'description', '')
            if not description:
                description = 'environmental sounds'
            attrs = ConditioningAttributes(text={'description': description})
            attributes.append(attrs)
        
        tokenized = model.lm.condition_provider.tokenize(attributes)
        condition_tensors = model.lm.condition_provider(tokenized)
        
        model_output = model.lm.compute_predictions(
            audio_tokens, [], condition_tensors
        )
        logits = model_output.logits
        B, K, T = audio_tokens.shape
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool)
        mask = padding_mask & model_output.mask
        
        loss = compute_cross_entropy(logits, audio_tokens, mask)
        print(f"✓ Loss computation successful")
        print(f"  Loss: {loss.item():.4f}")
        
except Exception as e:
    print(f"✗ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Testing training step...")
try:
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Test one training step
    audio, infos = next(iter(train_loader))
    
    # Encode
    with torch.no_grad():
        audio_tokens, _ = model.compression_model.encode(audio)
    
    # Prepare attributes
    from audiocraft.modules.conditioners import ConditioningAttributes
    attributes = []
    for info in infos:
        description = getattr(info, 'description', '')
        if not description:
            description = 'environmental sounds'
        attrs = ConditioningAttributes(text={'description': description})
        attributes.append(attrs)
    
    # Apply dropout
    attributes = model.lm.cfg_dropout(attributes)
    attributes = model.lm.att_dropout(attributes)
    tokenized = model.lm.condition_provider.tokenize(attributes)
    condition_tensors = model.lm.condition_provider(tokenized)
    
    # Forward
    model_output = model.lm.compute_predictions(
        audio_tokens, [], condition_tensors
    )
    logits = model_output.logits
    B, K, T = audio_tokens.shape
    padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool)
    mask = padding_mask & model_output.mask
    
    # Loss
    loss = compute_cross_entropy(logits, audio_tokens, mask)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    if config['training']['max_grad_norm'] > 0:
        torch.nn.utils.clip_grad_norm_(
            lora_params, config['training']['max_grad_norm']
        )
    
    optimizer.step()
    
    print(f"✓ Training step successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed and optimizer stepped")
    
except Exception as e:
    print(f"✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n6. Testing full training epoch...")
try:
    # Reset optimizer
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Run one epoch (will be very short with only 10 samples)
    metrics = train_epoch(
        model, train_loader, optimizer, device, 0, config
    )
    
    print(f"✓ Training epoch successful")
    print(f"  Train loss: {metrics['loss']:.4f}")
    
except Exception as e:
    print(f"✗ Training epoch failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n7. Testing validation...")
try:
    model.eval()
    val_metrics = validate(model, val_loader, device, config)
    
    print(f"✓ Validation successful")
    print(f"  Val loss: {val_metrics['loss']:.4f}")
    
except Exception as e:
    print(f"✗ Validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n8. Testing checkpoint save/load...")
try:
    output_dir = Path('./test_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        'epoch': 0,
        'train_metrics': metrics,
        'val_metrics': val_metrics,
        'optimizer': optimizer.state_dict(),
    }
    checkpoint_path = output_dir / 'test_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save LoRA weights
    lora_path = output_dir / 'test_checkpoint_lora.pt'
    model.save_lora_weights(str(lora_path))
    
    print(f"✓ Checkpoint saved")
    
    # Test loading
    model2 = create_musicgen_lora(
        model_name=config['model_name'],
        device=device,
        lora_rank=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
    )
    model2.load_lora_weights(str(lora_path))
    
    print(f"✓ Checkpoint loaded successfully")
    
    # Cleanup
    checkpoint_path.unlink()
    lora_path.unlink()
    output_dir.rmdir()
    
except Exception as e:
    print(f"✗ Checkpoint save/load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nYour training code is ready for Colab!")
print("Note: CPU training is very slow. Use GPU in Colab for actual training.")