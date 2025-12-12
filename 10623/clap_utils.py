# CLAP utilities for evaluation

import torch
import torch.nn.functional as F
from typing import List, Optional
import numpy as np

import laion_clap


class CLAPEvaluator:
    """CLAP evaluator for text-audio similarity."""
    def __init__(
        self,
        model_name: str = '630k',
        device: Optional[str] = None,
    ):  
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        # Try loading model (different API versions)
        try:
            self.model.load_ckpt(model_name=model_name)
        except TypeError:
            try:
                self.model.load_ckpt()
            except Exception as e:
                print(f"Warning: Could not load CLAP model, trying default...")
                self.model.load_ckpt()
        self.model.eval()
    
    @torch.no_grad()
    def compute_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Compute text embeddings."""
        text_embeddings = self.model.get_text_embedding(texts)
        return torch.tensor(text_embeddings, device=self.device)
    
    @torch.no_grad()
    def compute_audio_embeddings(self, audio: torch.Tensor, sample_rate: int = 32000) -> torch.Tensor:
        """Compute audio embeddings (resamples to 48kHz if needed)."""
        # Ensure audio is on correct device and format
        if audio.dim() == 3:
            # [B, C, T] -> [B, T] (take first channel if stereo)
            if audio.shape[1] > 1:
                audio = audio[:, 0, :]
            else:
                audio = audio.squeeze(1)
        
        # Resample to 48kHz if needed
        if sample_rate != 48000:
            from torchaudio.transforms import Resample
            resampler = Resample(sample_rate, 48000).to(audio.device)
            audio = resampler(audio)
        
        # Try different API versions
        try:
            audio_embeddings = self.model.get_audio_embedding_from_data(
                audio, use_tensor=True
            )
            if isinstance(audio_embeddings, torch.Tensor):
                return audio_embeddings.to(self.device)
            else:
                return torch.tensor(audio_embeddings, device=self.device)
        except (TypeError, AttributeError):
            audio_np = audio.cpu().numpy()
            try:
                audio_embeddings = self.model.get_audio_embedding_from_data(audio_np)
                return torch.tensor(audio_embeddings, device=self.device)
            except Exception as e:
                audio_list = [audio_np[i] for i in range(audio_np.shape[0])]
                audio_embeddings = self.model.get_audio_embedding_from_data(audio_list)
                return torch.tensor(audio_embeddings, device=self.device)
    
    @torch.no_grad()
    def compute_similarity(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between text and audio embeddings."""
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(text_embeddings, audio_embeddings.t())
        
        return similarity
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        texts: List[str],
        audio: torch.Tensor,
        sample_rate: int = 32000,
    ) -> dict:
        """Evaluate a batch of text-audio pairs."""
        text_embeddings = self.compute_text_embeddings(texts)
        audio_embeddings = self.compute_audio_embeddings(audio, sample_rate=sample_rate)
        
        similarity = self.compute_similarity(text_embeddings, audio_embeddings)
        
        # Diagonal elements are the matching pairs
        matching_similarities = torch.diag(similarity)
        
        # Compute metrics
        mean_similarity = matching_similarities.mean().item()
        std_similarity = matching_similarities.std().item()
        
        # Top-1 accuracy (for retrieval task)
        # For each text, find the most similar audio
        top1_accuracy = 0.0
        for i in range(len(texts)):
            top_audio_idx = similarity[i].argmax().item()
            if top_audio_idx == i:
                top1_accuracy += 1.0
        top1_accuracy /= len(texts)
        
        return {
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'top1_accuracy': top1_accuracy,
            'similarity_matrix': similarity.cpu().numpy(),
            'matching_similarities': matching_similarities.cpu().numpy(),
        }


def compute_clap_similarity(
    texts: List[str],
    audio: torch.Tensor,
    sample_rate: int = 32000,
    device: Optional[str] = None,
) -> float:
    """Compute CLAP similarity (convenience function)."""
    evaluator = CLAPEvaluator(device=device)
    results = evaluator.evaluate_batch(texts, audio, sample_rate=sample_rate)
    return results['mean_similarity']

