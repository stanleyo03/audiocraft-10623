# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CLAP (Contrastive Language-Audio Pretraining) utilities for evaluation.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    print("Warning: laion-clap not available. CLAP evaluation will not work.")


class CLAPEvaluator:
    """CLAP evaluator for text-audio similarity.
    
    Args:
        model_name: CLAP model name (default: '630k')
        device: Device to run evaluation on
    """
    def __init__(
        self,
        model_name: str = '630k',
        device: Optional[str] = None,
    ):
        if not CLAP_AVAILABLE:
            raise ImportError(
                "laion-clap is required for CLAP evaluation. "
                "Install with: pip install laion-clap"
            )
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        self.model.load_ckpt(model_name=model_name)
        self.model.eval()
    
    @torch.no_grad()
    def compute_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Compute text embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Text embeddings [N, D]
        """
        text_embeddings = self.model.get_text_embedding(texts)
        return torch.tensor(text_embeddings, device=self.device)
    
    @torch.no_grad()
    def compute_audio_embeddings(self, audio: torch.Tensor, sample_rate: int = 32000) -> torch.Tensor:
        """Compute audio embeddings.
        
        Args:
            audio: Audio waveforms [B, C, T] or [B, T]
            sample_rate: Sample rate of audio
            
        Returns:
            Audio embeddings [B, D]
        """
        # Ensure audio is on correct device and format
        if audio.dim() == 3:
            # [B, C, T] -> [B, T] (take first channel if stereo)
            if audio.shape[1] > 1:
                audio = audio[:, 0, :]
            else:
                audio = audio.squeeze(1)
        
        audio_np = audio.cpu().numpy()
        audio_embeddings = self.model.get_audio_embedding_from_data(
            audio_np, sample_rate=sample_rate
        )
        return torch.tensor(audio_embeddings, device=self.device)
    
    @torch.no_grad()
    def compute_similarity(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between text and audio embeddings.
        
        Args:
            text_embeddings: Text embeddings [N, D]
            audio_embeddings: Audio embeddings [M, D]
            
        Returns:
            Similarity matrix [N, M]
        """
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
        """Evaluate a batch of text-audio pairs.
        
        Args:
            texts: List of text descriptions
            audio: Audio waveforms [B, C, T] or [B, T]
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with similarity scores and metrics
        """
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
    """Compute CLAP similarity between texts and audio.
    
    Convenience function for single batch evaluation.
    
    Args:
        texts: List of text descriptions
        audio: Audio waveforms [B, C, T] or [B, T]
        sample_rate: Sample rate of audio
        device: Device to run on
        
    Returns:
        Mean similarity score
    """
    evaluator = CLAPEvaluator(device=device)
    results = evaluator.evaluate_batch(texts, audio, sample_rate=sample_rate)
    return results['mean_similarity']

