"""
Utilities for injecting spurious correlations into datasets for diagnostic testing.
"""
import torch
from torch.utils.data import Dataset
import random
from PIL import Image, ImageDraw


class SpuriousPCAM(Dataset):
    """
    Wraps a PCAM-like dataset to inject a spurious artifact (black square).
    
    Modes:
    - Training (spurious correlation): Inject ONLY into positive class (e.g., 50% of positives).
    - Analysis (balanced artifact): Inject into BOTH classes equally to test model sensitivity.
    """
    def __init__(self, base_dataset, artifact_probs=None, square_size=30, seed=42):
        """
        Args:
            base_dataset: The original dataset (returns (image, label)).
            artifact_probs: Dict mapping label (int) to probability of injection.
                            Example (Training): {0: 0.0, 1: 0.5} -> Correlation
                            Example (Analysis): {0: 0.5, 1: 0.5} -> No Correlation (Balanced)
            square_size: Size of the black square in pixels.
            seed: Random seed for reproducibility.
        """
        self.base_dataset = base_dataset
        self.artifact_probs = artifact_probs if artifact_probs is not None else {0: 0.0, 1: 0.5}
        self.square_size = square_size
        self.rng = random.Random(seed)
        
        # Pre-generate decisions for reproducibility
        self._inject_decisions = self._precompute_decisions()

    def _precompute_decisions(self):
        """Pre-compute injection decisions for reproducibility."""
        decisions = []
        for idx in range(len(self.base_dataset)):
            decisions.append(self.rng.random())
        return decisions

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 1. Get original sample
        sample = self.base_dataset[idx]
        
        # Handle tuple (torchvision) vs dict (HuggingFace)
        if isinstance(sample, (tuple, list)):
            img, label = sample[0], sample[1]
        elif isinstance(sample, dict):
            img = sample['image']
            label = sample['label']
        else:
            raise ValueError(f"Unknown sample type: {type(sample)}")

        # 2. Determine injection probability based on label
        # Ensure label is int for lookup
        label_val = label.item() if isinstance(label, torch.Tensor) else int(label)
        prob = self.artifact_probs.get(label_val, 0.0)

        # 3. Inject Artifact using pre-computed decision
        if self._inject_decisions[idx] < prob:
            # Copy to avoid mutating shared cache if applicable
            if hasattr(img, 'copy'):
                img = img.copy()
            
            # Draw black square (top-left)
            if isinstance(img, Image.Image):
                draw = ImageDraw.Draw(img)
                draw.rectangle([(0, 0), (self.square_size, self.square_size)], fill='black')
        
        return img, label


class RotationSpuriousPCAM(Dataset):
    """
    Wraps a PCAM-like dataset to inject a spurious artifact (90-degree rotation).
    
    Modes:
    - Training (spurious correlation): Rotate ONLY positive class images (e.g., 50% of positives).
    - Analysis (balanced artifact): Rotate BOTH classes equally to test model sensitivity.
    """
    def __init__(self, base_dataset, artifact_probs=None, seed=42):
        """
        Args:
            base_dataset: The original dataset (returns (image, label)).
            artifact_probs: Dict mapping label (int) to probability of rotation.
                            Example (Training): {0: 0.0, 1: 0.5} -> Correlation
                            Example (Analysis): {0: 0.5, 1: 0.5} -> No Correlation (Balanced)
            seed: Random seed for reproducibility.
        """
        self.base_dataset = base_dataset
        self.artifact_probs = artifact_probs if artifact_probs is not None else {0: 0.0, 1: 0.5}
        self.rng = random.Random(seed)
        
        # Pre-generate decisions for reproducibility
        self._inject_decisions = self._precompute_decisions()

    def _precompute_decisions(self):
        """Pre-compute injection decisions for reproducibility."""
        decisions = []
        for idx in range(len(self.base_dataset)):
            decisions.append(self.rng.random())
        return decisions

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 1. Get original sample
        sample = self.base_dataset[idx]
        
        # Handle tuple (torchvision) vs dict (HuggingFace)
        if isinstance(sample, (tuple, list)):
            img, label = sample[0], sample[1]
        elif isinstance(sample, dict):
            img = sample['image']
            label = sample['label']
        else:
            raise ValueError(f"Unknown sample type: {type(sample)}")

        # 2. Determine injection probability based on label
        # Ensure label is int for lookup
        label_val = label.item() if isinstance(label, torch.Tensor) else int(label)
        prob = self.artifact_probs.get(label_val, 0.0)

        # 3. Inject Artifact (90-degree rotation) using pre-computed decision
        if self._inject_decisions[idx] < prob:
            if isinstance(img, Image.Image):
                img = img.transpose(Image.Transpose.ROTATE_90)
        
        return img, label

