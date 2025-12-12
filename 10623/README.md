# MusicGen Fine-tuning for Environmental Audio (10623 Project)

This directory contains code for fine-tuning MusicGen-small on the ESC-50 environmental sound dataset using LoRA (Low-Rank Adaptation) adapters.

## Overview

The project adapts Meta's pretrained MusicGen model to generate environmental-instrumental hybrid audio. We use LoRA to efficiently fine-tune the transformer layers while keeping the pretrained weights frozen.

## Setup

### 1. Download ESC-50 Dataset

Option A: Set environment variable and download manually:
```bash
export ESC50_ROOT=/path/to/ESC-50
# Download from https://github.com/karolpiczak/ESC-50
```

Option B: Download manually from https://github.com/karolpiczak/ESC-50

### 2. Install Dependencies

Ensure you have the required packages:
- PyTorch
- audiocraft (this repo)
- laion-clap (for CLAP evaluation)
- Other dependencies from requirements.txt

### 3. Configuration

Edit `config_esc50_lora.yaml` to adjust training parameters:
- Learning rate
- Batch size
- Number of epochs
- LoRA rank and alpha
- Output directories

## Usage

### Training

```bash
python train_musicgen_esc50.py --config config_esc50_lora.yaml
```

### Evaluation

```bash
python eval_musicgen_esc50.py --checkpoint path/to/checkpoint.pt --config config_esc50_lora.yaml
```

### Inference

```bash
python run_inference.py
```

## Project Structure

- `lora.py`: LoRA adapter implementation
- `musicgen_lora_model.py`: Wrapper to add LoRA to MusicGen model
- `esc50_dataset.py`: ESC-50 dataset loader
- `train_musicgen_esc50.py`: Training script
- `eval_musicgen_esc50.py`: Evaluation script
- `run_inference.py`: Inference script
- `captions.py`: Generate captions from ESC-50 labels
- `clap_utils.py`: CLAP similarity evaluation utilities
- `config_esc50_lora.yaml`: Training configuration
- `run_in_colab_v2.ipynb`: Notebook for running on Colab/Kaggle

## Dataset Format

ESC-50 should be organized as:
```
ESC-50/
  audio/
    1-100032-A-0.wav
    1-100038-A-14.wav
    ...
  meta/
    esc50.csv
```

The dataset loader automatically generates text captions from the category labels.

## Evaluation Metrics

- **CLAP Similarity**: Text-audio semantic similarity using CLAP embeddings
- **FAD**: Fréchet Audio Distance (perceptual similarity)
- **KL Divergence**: Token distribution comparison

## Notes

- The model uses MusicGen-small (300M parameters) as the base
- Only transformer layers are fine-tuned with LoRA
- EnCodec compression model remains frozen
- Training uses EnCodec tokens (discrete representations)

