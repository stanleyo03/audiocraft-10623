
import sys
from pathlib import Path
import torch
import torchaudio

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from musicgen_lora_model import create_musicgen_lora

# Configuration
CHECKPOINT_PATH = 'test_outputs/test_checkpoint_lora.pt' 
MODEL_NAME = 'facebook/musicgen-small'
LORA_RANK = 8
LORA_ALPHA = 16.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prompts to generate
# prompts = ["soft piano with gentle rain",
# "calming synth under forest ambience",
# "acoustic guitar with ocean waves",
# "drums with echoes",
# "lo-fi piano mixed with city street noise",
# "slow violin over crackling fire",
# "relaxing flute with birds chirping",
# "clarinet in the wind",
# "electronic synth with thunder",
# "windchimes with nighttime crickets"]

prompts = ['chirping birds in the rain', 'water drops', 'crackling fire with crying baby', 'thunderstorm', 'dog barks after door wood knock', 'crow flushing a toilet', 'clapping sounds', 'breathing sounds while typing on keyboard', 'train with a siren']


print('Loading model...')
model = create_musicgen_lora(
    model_name=MODEL_NAME,
    device=DEVICE,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
)

# Load LoRA weights
checkpoint_path = Path(CHECKPOINT_PATH)
if checkpoint_path.exists():
    print(f'Loading LoRA weights from {CHECKPOINT_PATH}...')
    model.load_lora_weights(str(checkpoint_path))
    print('LoRA weights loaded')
else:
    print("Checkpoint not found")

model.set_generation_params(duration=10.0, use_sampling=True, top_k=250)

# Generate audio
for i, prompt in enumerate(prompts):
    print(f'\nGenerating: {prompt}')
    with torch.no_grad():
        wav = model.generate([prompt])
    
    output_file = f'test_outputs/wavs/generated_{i:02d}_{prompt.replace(" ", "_")[:30]}.wav'
    torchaudio.save(output_file, wav[0].cpu(), model.sample_rate)
    print(f'Saved: {output_file}')

