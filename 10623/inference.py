#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference script for fine-tuned MusicGen model.
Loads LoRA weights and generates audio from text prompts.
"""

import argparse
import sys
from pathlib import Path
import torch
import torchaudio

try:
    from .musicgen_lora_model import create_musicgen_lora
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent))
    from musicgen_lora_model import create_musicgen_lora


def main():
    parser = argparse.ArgumentParser(description='Generate audio with fine-tuned MusicGen')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to LoRA checkpoint (.pt file)')
    parser.add_argument('--prompts', type=str, nargs='+', required=True,
                        help='Text prompts for generation')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='Output directory for generated audio')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration of generated audio in seconds')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--model_name', type=str, default='facebook/musicgen-small',
                        help='Base MusicGen model name')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank (must match training)')
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                        help='LoRA alpha (must match training)')
    parser.add_argument('--use_sampling', action='store_true', default=True,
                        help='Use sampling for generation')
    parser.add_argument('--top_k', type=int, default=250,
                        help='Top-k sampling parameter')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = create_musicgen_lora(
        model_name=args.model_name,
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
    )
    print("✓ Model loaded")
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {args.checkpoint}")
    model.load_lora_weights(args.checkpoint)
    print("✓ LoRA weights loaded")
    
    # Set generation parameters
    model.set_generation_params(
        duration=args.duration,
        use_sampling=args.use_sampling,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate audio for each prompt
    print(f"\nGenerating audio for {len(args.prompts)} prompt(s)...")
    for i, prompt in enumerate(args.prompts):
        print(f"\n[{i+1}/{len(args.prompts)}] Generating: '{prompt}'")
        
        with torch.no_grad():
            audio = model.generate([prompt])
        
        # Save audio
        output_path = output_dir / f"generated_{i:03d}_{prompt.replace(' ', '_')[:50]}.wav"
        torchaudio.save(
            str(output_path),
            audio[0].cpu(),
            sample_rate=model.sample_rate
        )
        print(f"✓ Saved: {output_path}")
    
    print(f"\n✓ Generation complete! Audio saved to: {output_dir}")


if __name__ == '__main__':
    main()

