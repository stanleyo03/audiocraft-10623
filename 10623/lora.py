# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LoRA (Low-Rank Adaptation) implementation for fine-tuning transformer layers.
Based on: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import typing as tp


class LoRALinear(nn.Module):
    """LoRA adapter for a linear layer.
    
    Wraps a linear layer and adds low-rank adaptation:
    output = (W + BA) @ x
    
    where B and A are low-rank matrices with rank r << min(in_features, out_features).
    
    Args:
        linear_layer: The original linear layer to adapt
        rank: Rank of the low-rank adaptation (r)
        alpha: Scaling factor for LoRA weights (typically rank or 2*rank)
        dropout: Dropout probability for LoRA weights
    """
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA weights
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # Initialize LoRA weights
        # lora_A: random initialization with small values
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        # lora_B: zeros so initial output is zero (delta = 0 at start)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Ensure no NaN/Inf in initialization
        assert not torch.isnan(self.lora_A).any(), "NaN in lora_A initialization"
        assert not torch.isinf(self.lora_A).any(), "Inf in lora_A initialization"
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        output = self.linear(x)
        
        # LoRA adaptation: BA @ x
        # lora_A: [rank, in_features], we need [in_features, rank] for matmul
        # lora_B: [out_features, rank], we need [rank, out_features] for matmul
        x_dropout = self.dropout(x)
        # x @ lora_A.t() = x @ [in_features, rank] = [..., rank]
        lora_output = x_dropout @ self.lora_A.t()  # [..., rank]
        # lora_output @ lora_B.t() = [..., rank] @ [rank, out_features] = [..., out_features]
        lora_output = lora_output @ self.lora_B.t()  # [..., out_features]
        
        # Scale and add
        output = output + self.scaling * lora_output
        return output
    
    def merge_weights(self):
        """Merge LoRA weights into the original linear layer (for inference)."""
        with torch.no_grad():
            delta_W = self.scaling * (self.lora_B @ self.lora_A)
            self.linear.weight.data += delta_W.t()
    
    def unmerge_weights(self):
        """Unmerge LoRA weights (restore original weights)."""
        with torch.no_grad():
            delta_W = self.scaling * (self.lora_B @ self.lora_A)
            self.linear.weight.data -= delta_W.t()


def apply_lora_to_linear(
    linear_layer: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRALinear:
    """Apply LoRA to a linear layer.
    
    Args:
        linear_layer: The linear layer to wrap
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability
        
    Returns:
        LoRALinear wrapper
    """
    return LoRALinear(linear_layer, rank=rank, alpha=alpha, dropout=dropout)


def get_lora_parameters(model: tp.Union[nn.Module, tp.Any]) -> list:
    """Get all LoRA parameters that should be trained.
    
    Args:
        model: Model containing LoRA adapters (can be nn.Module or MusicGenLoRA)
        
    Returns:
        List of LoRA parameters (lora_A and lora_B)
    """
    lora_params = []
    
    # Handle MusicGenLoRA (which is not a nn.Module but has a .lm attribute)
    if hasattr(model, 'lm') and not isinstance(model, nn.Module):
        # This is likely a MusicGenLoRA or similar wrapper
        for module in model.lm.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.lora_A, module.lora_B])
    else:
        # Standard nn.Module
        for module in model.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.lora_A, module.lora_B])
    
    return lora_params

