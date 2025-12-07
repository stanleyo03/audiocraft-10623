# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ESC-50 dataset loader for MusicGen fine-tuning.
"""

import os
import csv
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import torch
import torch.utils.data

from audiocraft.data.audio_dataset import AudioDataset, AudioMeta
from audiocraft.data.info_audio_dataset import InfoAudioDataset, AudioInfo
from audiocraft.modules.conditioners import ConditioningAttributes

try:
    from .captions import get_caption_for_category, parse_esc50_filename, get_category_from_esc50_csv
except ImportError:
    from captions import get_caption_for_category, parse_esc50_filename, get_category_from_esc50_csv


class ESC50Dataset(InfoAudioDataset):
    """Dataset for ESC-50 environmental sounds.
    
    ESC-50 structure:
    ESC-50/
      audio/
        1-100032-A-0.wav
        ...
      meta/
        esc50.csv
    
    Args:
        root: Root directory of ESC-50 dataset
        split: 'train', 'valid', 'test', or None (use all)
        folds: List of folds to use (1-5). If None, uses all folds.
            For train/valid/test splits, typically use:
            - train: folds [1, 2, 3]
            - valid: fold [4]
            - test: fold [5]
        segment_duration: Duration of audio segments to load (None = full audio)
        sample_rate: Target sample rate
        channels: Number of channels (1 for mono)
        return_info: Whether to return metadata along with audio
    """
    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        folds: Optional[List[int]] = None,
        segment_duration: Optional[float] = None,
        sample_rate: int = 32000,
        channels: int = 1,
        return_info: bool = True,
        shuffle: bool = True,
        num_samples: int = 10000,
        **kwargs
    ):
        self.root = Path(root)
        self.audio_dir = self.root / 'audio'
        self.meta_file = self.root / 'meta' / 'esc50.csv'
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter by split/folds
        if split is not None or folds is not None:
            self.metadata = self._filter_by_split(self.metadata, split, folds)
        
        # Create AudioMeta list
        audio_meta_list = self._create_audio_meta_list()
        
        # Initialize parent class
        super().__init__(
            meta=audio_meta_list,
            segment_duration=segment_duration,
            sample_rate=sample_rate,
            channels=channels,
            return_info=return_info,
            shuffle=shuffle,
            num_samples=num_samples,
            **kwargs
        )
        
        # Store category mapping
        self.category_map = self._build_category_map()
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata from ESC-50 CSV file.
        
        Returns:
            List of metadata dictionaries
        """
        metadata = []
        
        if not self.meta_file.exists():
            raise FileNotFoundError(
                f"ESC-50 metadata file not found: {self.meta_file}\n"
                f"Please ensure ESC-50 is downloaded and extracted correctly."
            )
        
        with open(self.meta_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata.append({
                    'filename': row['filename'],
                    'fold': int(row['fold']),
                    'target': int(row['target']),
                    'category': row['category'],
                    'esc10': row.get('esc10', 'False') == 'True',
                })
        
        return metadata
    
    def _filter_by_split(
        self,
        metadata: List[Dict],
        split: Optional[str],
        folds: Optional[List[int]]
    ) -> List[Dict]:
        """Filter metadata by split or folds.
        
        Args:
            metadata: List of metadata dictionaries
            split: 'train', 'valid', 'test', or None
            folds: List of fold numbers (1-5) or None
            
        Returns:
            Filtered metadata
        """
        if folds is not None:
            metadata = [m for m in metadata if m['fold'] in folds]
        elif split == 'train':
            metadata = [m for m in metadata if m['fold'] in [1, 2, 3]]
        elif split == 'valid':
            metadata = [m for m in metadata if m['fold'] == 4]
        elif split == 'test':
            metadata = [m for m in metadata if m['fold'] == 5]
        
        return metadata
    
    def _create_audio_meta_list(self) -> List[AudioMeta]:
        """Create list of AudioMeta from metadata.
        
        Returns:
            List of AudioMeta objects
        """
        audio_meta_list = []
        
        for meta in self.metadata:
            audio_path = self.audio_dir / meta['filename']
            if not audio_path.exists():
                continue
            
            # Get audio info
            from audiocraft.data.audio import audio_info
            info = audio_info(str(audio_path))
            
            audio_meta = AudioMeta(
                path=str(audio_path),
                duration=info.duration,
                sample_rate=info.sample_rate,
            )
            audio_meta_list.append(audio_meta)
        
        return audio_meta_list
    
    def _build_category_map(self) -> Dict[str, str]:
        """Build mapping from filename to category.
        
        Returns:
            Dictionary mapping filename to category name
        """
        category_map = {}
        for meta in self.metadata:
            category_map[meta['filename']] = meta['category']
        return category_map
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, AudioInfo]:
        """Get item from dataset.
        
        Args:
            index: Dataset index
            
        Returns:
            Tuple of (waveform, AudioInfo)
        """
        wav, info = super().__getitem__(index)
        
        # Get filename from metadata
        audio_meta = self.meta[index]
        filename = Path(audio_meta.path).name
        
        # Get category and generate caption
        category = self.category_map.get(filename, '')
        caption = get_caption_for_category(category)
        
        # Create AudioInfo with caption
        audio_info = AudioInfo(
            meta=info.meta,
            seek_time=info.seek_time,
            n_frames=info.n_frames,
            total_frames=info.total_frames,
            sample_rate=info.sample_rate,
            channels=info.channels,
        )
        
        # Add description for conditioning
        audio_info.description = caption
        
        return wav, audio_info
    
    def to_condition_attributes(self, info: AudioInfo) -> ConditioningAttributes:
        """Convert AudioInfo to ConditioningAttributes for MusicGen.
        
        Args:
            info: AudioInfo object
            
        Returns:
            ConditioningAttributes
        """
        attrs = ConditioningAttributes()
        if hasattr(info, 'description') and info.description:
            attrs.text = {'description': info.description}
        return attrs


def create_esc50_dataloader(
    root: str,
    split: str = 'train',
    batch_size: int = 4,
    num_workers: int = 4,
    segment_duration: Optional[float] = None,
    sample_rate: int = 32000,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for ESC-50 dataset.
    
    Args:
        root: Root directory of ESC-50
        split: 'train', 'valid', or 'test'
        batch_size: Batch size
        num_workers: Number of worker processes
        segment_duration: Duration of segments (None = full audio)
        sample_rate: Target sample rate
        **kwargs: Additional arguments for ESC50Dataset
        
    Returns:
        DataLoader
    """
    dataset = ESC50Dataset(
        root=root,
        split=split,
        segment_duration=segment_duration,
        sample_rate=sample_rate,
        return_info=True,
        **kwargs
    )
    
    def collate_fn(batch):
        """Collate function for ESC-50 dataset."""
        waveforms = []
        infos = []
        for wav, info in batch:
            waveforms.append(wav)
            infos.append(info)
        
        # Stack waveforms
        max_len = max(w.shape[-1] for w in waveforms)
        padded_waveforms = []
        for w in waveforms:
            if w.shape[-1] < max_len:
                padding = torch.zeros(
                    w.shape[0], max_len - w.shape[-1],
                    device=w.device, dtype=w.dtype
                )
                w = torch.cat([w, padding], dim=-1)
            padded_waveforms.append(w)
        
        stacked_wav = torch.stack(padded_waveforms)
        
        return stacked_wav, infos
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == 'train'),
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return loader

