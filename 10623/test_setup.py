# Quick test script
import os
os.environ['ESC50_ROOT'] = "/mnt/c/Users/stano/Documents/CMU/Fall 2025/10623/audiocraft-10623/10623/ESC-50"

from musicgen_lora_model import create_musicgen_lora
from esc50_dataset import create_esc50_dataloader

# Test model loading
model = create_musicgen_lora('facebook/musicgen-small', device='cuda')
print("✓ Model loaded successfully")

# Test dataset
dataloader = create_esc50_dataloader(
    root=os.environ['ESC50_ROOT'],
    split='train',
    batch_size=2,
    num_workers=0,  # Use 0 for testing
)
print("✓ Dataset loaded successfully")

# Test one batch
audio, infos = next(iter(dataloader))
print(f"✓ Batch shape: {audio.shape}")
print(f"✓ Captions: {[getattr(info, 'description', '') for info in infos]}")