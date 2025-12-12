# Evaluation script for MusicGen fine-tuned on ESC-50

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
    audio, sr = torchaudio.load(str(file_path))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if target_length is not None:
        if audio.shape[1] < target_length:
            padding = target_length - audio.shape[1]
            audio = F.pad(audio, (0, padding))
        elif audio.shape[1] > target_length:
            audio = audio[:, :target_length]
    return audio




def compute_kl_divergence(real_tokens: torch.Tensor, gen_tokens: torch.Tensor, vocab_size: int = 2048) -> float:
    real_flat = real_tokens.view(-1).long()
    gen_flat = gen_tokens.view(-1).long()
    real_dist = torch.bincount(real_flat, minlength=vocab_size).float()
    real_dist = real_dist / real_dist.sum() + 1e-10
    gen_dist = torch.bincount(gen_flat, minlength=vocab_size).float()
    gen_dist = gen_dist / gen_dist.sum() + 1e-10
    kl = (real_dist * torch.log(real_dist / gen_dist)).sum()
    return kl.item()


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: str, config: dict, output_dir: Optional[Path] = None) -> Dict[str, float]:
    model.eval()
    clap_evaluator = None
    try:
        clap_evaluator = CLAPEvaluator(device=device)
    except Exception as e:
        print(f"CLAP init failed: {e}")
    
    all_clap_similarities = []
    all_kls = []
    
    num_samples = config.get('eval_num_samples', 100)
    gen_duration = config.get('gen_duration', 10.0)
    
    print(f"Evaluating on {num_samples} samples...")
    
    for batch_idx, (audio, infos) in enumerate(tqdm(dataloader, desc="Evaluation")):
        if batch_idx * dataloader.batch_size >= num_samples:
            break
        
        audio = audio.to(device)
        batch_size = audio.shape[0]
        
        texts = []
        for info in infos:
            description = getattr(info, 'description', '')
            if not description:
                description = 'environmental sounds'
            texts.append(description)
        
        attributes = []
        for info in infos:
            from audiocraft.modules.conditioners import ConditioningAttributes
            description = getattr(info, 'description', '')
            attrs = ConditioningAttributes(text={'description': description})
            attributes.append(attrs)
        
        total_gen_len = int(gen_duration * model.compression_model.frame_rate)
        gen_tokens = model.lm.generate(None, attributes, max_gen_len=total_gen_len,
            use_sampling=True, temp=1.0, top_k=250, top_p=0.0)
        gen_audio = model.compression_model.decode(gen_tokens, None)
        
        # Compute CLAP
        if clap_evaluator:
            try:
                results = clap_evaluator.evaluate_batch(
                    texts, gen_audio, sample_rate=32000
                )
                all_clap_similarities.append(results['mean_similarity'])
            except Exception as e:
                print(f"CLAP failed: {e}")
        
        # Compute KL divergence
        try:
            real_tokens, _ = model.compression_model.encode(audio)
            kl = compute_kl_divergence(real_tokens, gen_tokens)
            all_kls.append(kl)
        except Exception as e:
            print(f"Warning: KL divergence computation failed: {e}")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for i in range(batch_size):
                idx = batch_idx * dataloader.batch_size + i
                if idx < num_samples:
                    torchaudio.save(str(output_dir / f"gen_{idx:04d}.wav"), gen_audio[i].cpu(), 32000)
                    torchaudio.save(str(output_dir / f"real_{idx:04d}.wav"), audio[i].cpu(), 32000)
                    with open(output_dir / f"text_{idx:04d}.txt", 'w') as f:
                        f.write(texts[i])
    
    metrics = {}
    if all_clap_similarities:
        metrics['clap_similarity'] = np.mean(all_clap_similarities)
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
    
    clap_evaluator = None
    try:
        clap_evaluator = CLAPEvaluator(device=device)
    except Exception as e:
        print(f"CLAP init failed: {e}")
    
    all_ground_truth_audio = []
    all_pretrained_audio = []
    all_finetuned_audio = []
    all_prompts = []
    per_prompt_metrics = []
    
    target_sr = 32000
    max_length = int(gen_duration * target_sr)
    
    print(f"Evaluating {len(prompts)} prompts...")
    
    for prompt in tqdm(prompts, desc="Generating"):
        audio_file = prompt_to_file.get(prompt)
        if not audio_file:
            continue
        
        gt_path = ground_truth_dir / audio_file
        if not gt_path.exists():
            continue
        
        gt_audio = load_audio_file(gt_path, target_sr=target_sr, target_length=max_length)
        gt_audio = gt_audio.to(device)
        
        # Generate with pretrained
        pretrained_model.lm.eval()
        pretrained_model.compression_model.eval()
        pretrained_model.set_generation_params(duration=gen_duration, use_sampling=True, top_k=250)
        pretrained_wav = pretrained_model.generate([prompt])
        pretrained_audio = pretrained_wav[0].to(device)
        
        if pretrained_audio.shape[1] > max_length:
            pretrained_audio = pretrained_audio[:, :max_length]
        elif pretrained_audio.shape[1] < max_length:
            padding = max_length - pretrained_audio.shape[1]
            pretrained_audio = F.pad(pretrained_audio, (0, padding))
        
        # Generate with fine-tuned
        finetuned_model.eval()
        finetuned_model.set_generation_params(duration=gen_duration, use_sampling=True, top_k=250)
        finetuned_wav = finetuned_model.generate([prompt])
        finetuned_audio = finetuned_wav[0].to(device)
        
        if finetuned_audio.shape[1] > max_length:
            finetuned_audio = finetuned_audio[:, :max_length]
        elif finetuned_audio.shape[1] < max_length:
            padding = max_length - finetuned_audio.shape[1]
            finetuned_audio = F.pad(finetuned_audio, (0, padding))
        
        all_ground_truth_audio.append(gt_audio.cpu())
        all_pretrained_audio.append(pretrained_audio.cpu())
        all_finetuned_audio.append(finetuned_audio.cpu())
        all_prompts.append(prompt)
        
        # Save samples
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            idx = len(all_prompts) - 1
            base_name = audio_file.replace('.wav', '')
            torchaudio.save(str(output_dir / f"{base_name}_ground_truth.wav"), gt_audio.cpu(), target_sr)
            torchaudio.save(str(output_dir / f"{base_name}_pretrained.wav"), pretrained_audio.cpu(), target_sr)
            torchaudio.save(str(output_dir / f"{base_name}_finetuned.wav"), finetuned_audio.cpu(), target_sr)
            with open(output_dir / f"{base_name}_prompt.txt", 'w') as f:
                f.write(prompt)
    
    gt_audio_batch = torch.stack(all_ground_truth_audio)
    pretrained_audio_batch = torch.stack(all_pretrained_audio)
    finetuned_audio_batch = torch.stack(all_finetuned_audio)
    
    if pretrained_audio_batch.shape[1] > 1:
        pretrained_audio_batch = pretrained_audio_batch.mean(dim=1, keepdim=True)
    if finetuned_audio_batch.shape[1] > 1:
        finetuned_audio_batch = finetuned_audio_batch.mean(dim=1, keepdim=True)
    
    metrics = {}
    per_prompt_metrics = [{'prompt': prompt, 'prompt_idx': i} for i, prompt in enumerate(all_prompts)]
    
    # CLAP
    if clap_evaluator:
        print("\nComputing CLAP...")
        try:
            for i, prompt in enumerate(all_prompts):
                gt_clap = clap_evaluator.evaluate_batch([prompt], gt_audio_batch[i:i+1], sample_rate=target_sr)
                per_prompt_metrics[i]['clap_ground_truth'] = gt_clap['mean_similarity']
                
                pretrained_clap = clap_evaluator.evaluate_batch([prompt], pretrained_audio_batch[i:i+1], sample_rate=target_sr)
                per_prompt_metrics[i]['clap_pretrained'] = pretrained_clap['mean_similarity']
                
                finetuned_clap = clap_evaluator.evaluate_batch([prompt], finetuned_audio_batch[i:i+1], sample_rate=target_sr)
                per_prompt_metrics[i]['clap_finetuned'] = finetuned_clap['mean_similarity']
            
            metrics['clap_ground_truth'] = np.mean([m['clap_ground_truth'] for m in per_prompt_metrics])
            metrics['clap_pretrained'] = np.mean([m['clap_pretrained'] for m in per_prompt_metrics])
            metrics['clap_finetuned'] = np.mean([m['clap_finetuned'] for m in per_prompt_metrics])
            
            print(f"Ground truth CLAP: {metrics['clap_ground_truth']:.4f}")
            print(f"Pretrained CLAP: {metrics['clap_pretrained']:.4f}")
            print(f"Fine-tuned CLAP: {metrics['clap_finetuned']:.4f}")
        except Exception as e:
            print(f"CLAP failed: {e}")
    
    # KL divergence
    print("\nComputing KL divergence...")
    pretrained_kls = []
    finetuned_kls = []
    compression_model = finetuned_model.compression_model
    
    for i in range(len(all_prompts)):
        try:
            gt_tokens, _ = compression_model.encode(gt_audio_batch[i:i+1].to(device))
            pretrained_tokens, _ = compression_model.encode(pretrained_audio_batch[i:i+1].to(device))
            kl_pretrained = compute_kl_divergence(gt_tokens, pretrained_tokens)
            pretrained_kls.append(kl_pretrained)
            if i < len(per_prompt_metrics):
                per_prompt_metrics[i]['kl_pretrained'] = kl_pretrained
            
            finetuned_tokens, _ = compression_model.encode(finetuned_audio_batch[i:i+1].to(device))
            kl_finetuned = compute_kl_divergence(gt_tokens, finetuned_tokens)
            finetuned_kls.append(kl_finetuned)
            if i < len(per_prompt_metrics):
                per_prompt_metrics[i]['kl_finetuned'] = kl_finetuned
        except Exception as e:
            print(f"KL failed for sample {i}: {e}")
    
    if pretrained_kls:
        metrics['kl_pretrained'] = np.mean(pretrained_kls)
        print(f"Pretrained KL: {metrics['kl_pretrained']:.4f}")
    if finetuned_kls:
        metrics['kl_finetuned'] = np.mean(finetuned_kls)
        print(f"Fine-tuned KL: {metrics['kl_finetuned']:.4f}")
    
    # Save CSV
    if output_dir and per_prompt_metrics:
        import csv
        csv_path = output_dir / 'per_prompt_metrics.csv'
        fieldnames = ['prompt_idx', 'prompt', 'clap_ground_truth', 'clap_pretrained', 'clap_finetuned',
                     'kl_pretrained', 'kl_finetuned']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in per_prompt_metrics:
                row = {field: m.get(field, '') for field in fieldnames}
                writer.writerow(row)
        print(f"\nPer-prompt metrics saved to {csv_path}")
    
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

