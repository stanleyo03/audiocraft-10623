import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import typing as tp


class LoRALinear(nn.Module):
    """LoRA adapter wrapper for linear layers.
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
        
        # Get device from the wrapped layer
        device = next(linear_layer.parameters()).device
        
        # Initialize LoRA weights
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)
        
        # Make sure LoRA params are on the right device
        device = x.device
        if self.lora_A.device != device:
            self.lora_A.data = self.lora_A.data.to(device)
        if self.lora_B.device != device:
            self.lora_B.data = self.lora_B.data.to(device)
        
        # Apply LoRA: BA @ x
        x_dropout = self.dropout(x)
        lora_output = x_dropout @ self.lora_A.t()
        lora_output = lora_output @ self.lora_B.t()
        
        output = output + self.scaling * lora_output
        return output
    
    def merge_weights(self):
        """Merge LoRA weights into base weights for faster inference."""
        with torch.no_grad():
            delta_W = self.scaling * (self.lora_B @ self.lora_A)
            self.linear.weight.data += delta_W.t()
    
    def unmerge_weights(self):
        """Restore original weights."""
        with torch.no_grad():
            delta_W = self.scaling * (self.lora_B @ self.lora_A)
            self.linear.weight.data -= delta_W.t()


def apply_lora_to_linear(
    linear_layer: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> LoRALinear:
    """Wrap a linear layer with LoRA."""
    return LoRALinear(linear_layer, rank=rank, alpha=alpha, dropout=dropout)


def get_lora_parameters(model: tp.Union[nn.Module, tp.Any]) -> list:
    """Get all LoRA parameters for training."""
    lora_params = []
    
    # Handle MusicGenLoRA wrapper
    if hasattr(model, 'lm') and not isinstance(model, nn.Module):
        for module in model.lm.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.lora_A, module.lora_B])
    else:
        # Standard nn.Module
        for module in model.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend([module.lora_A, module.lora_B])
    
    return lora_params

