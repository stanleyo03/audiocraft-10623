# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper to add LoRA adapters to MusicGen model.
Applies LoRA to transformer layers in the language model.
"""

import typing as tp
import torch
import torch.nn as nn

from audiocraft.models.musicgen import MusicGen
from audiocraft.models.lm import LMModel

try:
    from .lora import LoRALinear, apply_lora_to_linear, get_lora_parameters
except ImportError:
    from lora import LoRALinear, apply_lora_to_linear, get_lora_parameters


def add_lora_to_transformer_layer(
    layer: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: tp.Optional[tp.List[str]] = None,
) -> None:
    """Add LoRA adapters to a transformer layer.
    
    Args:
        layer: Transformer layer (StreamingTransformerLayer)
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability
        target_modules: List of module names to apply LoRA to.
            If None, applies to attention and feedforward layers.
    """
    if target_modules is None:
        target_modules = ['self_attn', 'linear1', 'linear2']
    
    # Apply LoRA to attention layers
    if 'self_attn' in target_modules and hasattr(layer, 'self_attn'):
        attn = layer.self_attn
        # Check if it's a StreamingMultiheadAttention
        if hasattr(attn, 'in_proj_weight'):
            # For custom attention, we need to handle in_proj_weight differently
            # For now, we'll apply LoRA to the out_proj if it exists
            if hasattr(attn, 'out_proj'):
                attn.out_proj = apply_lora_to_linear(
                    attn.out_proj, rank=rank, alpha=alpha, dropout=dropout
                )
        elif hasattr(attn, 'out_proj'):
            attn.out_proj = apply_lora_to_linear(
                attn.out_proj, rank=rank, alpha=alpha, dropout=dropout
            )
    
    # Apply LoRA to feedforward layers
    if 'linear1' in target_modules and hasattr(layer, 'linear1'):
        layer.linear1 = apply_lora_to_linear(
            layer.linear1, rank=rank, alpha=alpha, dropout=dropout
        )
    
    if 'linear2' in target_modules and hasattr(layer, 'linear2'):
        layer.linear2 = apply_lora_to_linear(
            layer.linear2, rank=rank, alpha=alpha, dropout=dropout
        )


def add_lora_to_lm_model(
    lm_model: LMModel,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: tp.Optional[tp.List[str]] = None,
) -> LMModel:
    """Add LoRA adapters to all transformer layers in an LMModel.
    
    Args:
        lm_model: The language model to add LoRA to
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout probability
        target_modules: List of module names to apply LoRA to
        
    Returns:
        The same model with LoRA adapters added
    """
    transformer = lm_model.transformer
    
    # Apply LoRA to each transformer layer
    if hasattr(transformer, 'layers'):
        for layer in transformer.layers:
            add_lora_to_transformer_layer(
                layer, rank=rank, alpha=alpha, dropout=dropout,
                target_modules=target_modules
            )
    elif hasattr(transformer, 'encoder'):
        # Alternative structure
        for layer in transformer.encoder.layers:
            add_lora_to_transformer_layer(
                layer, rank=rank, alpha=alpha, dropout=dropout,
                target_modules=target_modules
            )
    
    return lm_model


class MusicGenLoRA(MusicGen):
    """MusicGen model with LoRA adapters for fine-tuning.
    
    This wrapper adds LoRA adapters to the transformer layers while keeping
    the pretrained weights frozen. Only LoRA parameters are trainable.
    
    Args:
        base_model: Pretrained MusicGen model
        lora_rank: Rank of LoRA adapters
        lora_alpha: Alpha scaling factor for LoRA
        lora_dropout: Dropout probability for LoRA
        target_modules: List of module names to apply LoRA to
    """
    def __init__(
        self,
        base_model: MusicGen,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: tp.Optional[tp.List[str]] = None,
    ):
        # Initialize with base model components
        super().__init__(
            name=base_model.name,
            compression_model=base_model.compression_model,
            lm=base_model.lm,
            max_duration=base_model.max_duration,
        )
        
        # Add LoRA to the language model
        self.lm = add_lora_to_lm_model(
            self.lm,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
        )
        
        # Store LoRA config
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Freeze all non-LoRA parameters
        self._freeze_base_parameters()
    
    def train(self, mode: bool = True):
        """Set the model to training mode.
        
        Args:
            mode: If True, set to training mode. If False, set to eval mode.
        """
        # BaseGenModel is not a nn.Module, so we set submodules
        self.lm.train(mode)
        self.compression_model.eval()  # Always keep compression model in eval
        return self
    
    def eval(self):
        """Set the model to evaluation mode."""
        self.lm.eval()
        self.compression_model.eval()
        return self
    
    def _freeze_base_parameters(self):
        """Freeze all base model parameters except LoRA."""
        # BaseGenModel is not a nn.Module, so we need to freeze submodules
        # Freeze compression model
        for param in self.compression_model.parameters():
            param.requires_grad = False
        
        # Freeze language model (except LoRA)
        for param in self.lm.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA parameters
        for module in self.lm.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
    
    def get_lora_parameters(self):
        """Get all LoRA parameters for optimizer."""
        # BaseGenModel is not a nn.Module, so get parameters from lm
        return get_lora_parameters(self.lm)
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights to a file.
        
        Args:
            path: Path to save the LoRA weights
        """
        lora_state_dict = {}
        # BaseGenModel is not a nn.Module, so iterate over lm submodules
        for name, module in self.lm.named_modules():
            if isinstance(module, LoRALinear):
                lora_state_dict[f"lm.{name}.lora_A"] = module.lora_A
                lora_state_dict[f"lm.{name}.lora_B"] = module.lora_B
        
        torch.save(lora_state_dict, path)
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights from a file.
        
        Args:
            path: Path to load the LoRA weights from
        """
        lora_state_dict = torch.load(path, map_location=self.device)
        # BaseGenModel is not a nn.Module, so iterate over lm submodules
        for name, module in self.lm.named_modules():
            if isinstance(module, LoRALinear):
                # Try both with and without "lm." prefix for compatibility
                key_a = f"lm.{name}.lora_A"
                key_a_alt = f"{name}.lora_A"
                key_b = f"lm.{name}.lora_B"
                key_b_alt = f"{name}.lora_B"
                
                if key_a in lora_state_dict:
                    module.lora_A.data = lora_state_dict[key_a]
                elif key_a_alt in lora_state_dict:
                    module.lora_A.data = lora_state_dict[key_a_alt]
                
                if key_b in lora_state_dict:
                    module.lora_B.data = lora_state_dict[key_b]
                elif key_b_alt in lora_state_dict:
                    module.lora_B.data = lora_state_dict[key_b_alt]


def create_musicgen_lora(
    model_name: str = 'facebook/musicgen-small',
    device: tp.Optional[str] = None,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: tp.Optional[tp.List[str]] = None,
) -> MusicGenLoRA:
    """Create a MusicGen model with LoRA adapters.
    
    Args:
        model_name: Name of the pretrained MusicGen model
        device: Device to load the model on
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability
        target_modules: List of module names to apply LoRA to
        
    Returns:
        MusicGenLoRA model
    """
    # Load base model
    base_model = MusicGen.get_pretrained(model_name, device=device)
    
    # Wrap with LoRA
    model = MusicGenLoRA(
        base_model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    
    return model

