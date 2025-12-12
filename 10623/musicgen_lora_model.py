
# Wrapper to add LoRA adapters to MusicGen
# Applies LoRA to transformer layers

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
    """Add LoRA to transformer layer."""
    if target_modules is None:
        target_modules = ['self_attn', 'linear1', 'linear2']
    
    # Apply to attention
    if 'self_attn' in target_modules and hasattr(layer, 'self_attn'):
        attn = layer.self_attn
        if hasattr(attn, 'out_proj'):
            attn.out_proj = apply_lora_to_linear(
                attn.out_proj, rank=rank, alpha=alpha, dropout=dropout
            )
    
    # Apply to feedforward
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
    """Add LoRA to all transformer layers."""
    transformer = lm_model.transformer
    
    if hasattr(transformer, 'layers'):
        for layer in transformer.layers:
            add_lora_to_transformer_layer(
                layer, rank=rank, alpha=alpha, dropout=dropout,
                target_modules=target_modules
            )
    elif hasattr(transformer, 'encoder'):
        for layer in transformer.encoder.layers:
            add_lora_to_transformer_layer(
                layer, rank=rank, alpha=alpha, dropout=dropout,
                target_modules=target_modules
            )
    
    return lm_model


class MusicGenLoRA(MusicGen):
    """MusicGen with LoRA adapters for fine-tuning.
    
    Only LoRA parameters are trainable, base weights stay frozen.
    """
    def __init__(
        self,
        base_model: MusicGen,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: tp.Optional[tp.List[str]] = None,
    ):
        super().__init__(
            name=base_model.name,
            compression_model=base_model.compression_model,
            lm=base_model.lm,
            max_duration=base_model.max_duration,
        )
        
        # Add LoRA to transformer layers
        self.lm = add_lora_to_lm_model(
            self.lm,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
        )
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self._freeze_base_parameters()
    
    def train(self, mode: bool = True):
        self.lm.train(mode)
        self.compression_model.eval()  # Keep compression model frozen
        return self
    
    def eval(self):
        self.lm.eval()
        self.compression_model.eval()
        return self
    
    def _freeze_base_parameters(self):
        """Freeze base model, only train LoRA."""
        for param in self.compression_model.parameters():
            param.requires_grad = False
        
        for param in self.lm.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA
        for module in self.lm.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
    
    def to_device(self, device: tp.Union[str, torch.device]):
        """Move model to device."""
        if isinstance(device, str):
            device = torch.device(device)
        
        # Move compression model
        if hasattr(self.compression_model, 'to'):
            self.compression_model = self.compression_model.to(device)
        else:
            for param in self.compression_model.parameters():
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        
        # Move language model
        if hasattr(self.lm, 'to'):
            self.lm = self.lm.to(device)
        else:
            for param in self.lm.parameters():
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        
        # Move LoRA params
        for module in self.lm.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.data = module.lora_A.data.to(device)
                module.lora_B.data = module.lora_B.data.to(device)
        
        return self
    
    def get_lora_parameters(self):
        """Get LoRA parameters for optimizer."""
        return get_lora_parameters(self.lm)
    
    def save_lora_weights(self, path: str):
        """Save LoRA weights."""
        lora_state_dict = {}
        for name, module in self.lm.named_modules():
            if isinstance(module, LoRALinear):
                lora_state_dict[f"lm.{name}.lora_A"] = module.lora_A
                lora_state_dict[f"lm.{name}.lora_B"] = module.lora_B
        
        torch.save(lora_state_dict, path)
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        lora_state_dict = torch.load(path, map_location=self.device)
        for name, module in self.lm.named_modules():
            if isinstance(module, LoRALinear):
                # Try with and without "lm." prefix
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
    """Create MusicGen model with LoRA."""
    base_model = MusicGen.get_pretrained(model_name, device=device)
    
    model = MusicGenLoRA(
        base_model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    
    if device is not None:
        model.to_device(device)
    
    return model

