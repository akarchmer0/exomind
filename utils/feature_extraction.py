import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

def extract_features(model, processor, dataset, device=None, batch_size=32):
    """
    Extracts representations and labels from a dataset using a given model and processor.

    Args:
        model: The loaded HuggingFace model (e.g., Phikon).
        processor: The loaded HuggingFace image processor.
        dataset: A dataset object. Can be:
                 - HuggingFace dataset (must have 'image' and 'label' columns/keys)
                 - PyTorch Dataset (must return (image, label) tuples)
        device: 'cuda' or 'cpu'. If None, detects automatically.
        batch_size: Batch size for processing.

    Returns:
        features: Tensor of shape (num_samples, feature_dim)
        labels: Tensor of shape (num_samples,)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    # Check if it's a Hugging Face dataset (dict-like access supports slicing)
    # A simple heuristic: check if it has 'features' attribute or if slicing returns a dict
    is_hf_dataset = False
    try:
        # HuggingFace datasets allow slicing and return dict of lists
        if hasattr(dataset, 'features') or isinstance(dataset[0], dict):
             is_hf_dataset = True
    except:
        pass # Fallback to PyTorch style

    if is_hf_dataset:
        iterator = range(0, len(dataset), batch_size)
        for i in tqdm(iterator, desc="Extracting features (HF)"):
            batch = dataset[i : i + batch_size]
            images = batch['image']
            labels = batch['label']
            
            _process_batch(model, processor, images, labels, all_features, all_labels, device)
    else:
        # Assume PyTorch Dataset style (returns (image, label))
        # Use DataLoader to handle batching
        def collate_fn(batch):
            # batch is list of tuples (img, label)
            images = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            return images, labels

        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
        
        for images, labels in tqdm(loader, desc="Extracting features (PyTorch)"):
            _process_batch(model, processor, images, labels, all_features, all_labels, device)

    # Concatenate all features and convert labels to tensor
    if not all_features:
        return torch.tensor([]), torch.tensor([])
        
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    return features_tensor, labels_tensor

def _process_batch(model, processor, images, labels, all_features, all_labels, device):
    """Helper to process a batch of images and labels."""
    # Convert images to PIL Images if needed, and ensure RGB format
    processed_images = []
    for img in images:
        # If it's a string path, load the image
        if isinstance(img, str):
            img = Image.open(img)
        
        # If it's a numpy array, convert to PIL Image
        if isinstance(img, np.ndarray):
            # Handle grayscale (2D) or RGB (3D) arrays
            if img.ndim == 2:
                img = Image.fromarray(img, mode='L').convert('RGB')
            elif img.ndim == 3:
                img = Image.fromarray(img)
            else:
                raise ValueError(f"Unsupported array shape: {img.shape}")
        
        # If it's already a PIL Image, ensure it's RGB
        if isinstance(img, Image.Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        processed_images.append(img)
    
    # processor returns a dict with 'pixel_values'
    inputs = processor(images=processed_images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Assuming ViT-like architecture, take the CLS token (first token)
        features = outputs.last_hidden_state[:, 0, :]
        
    all_features.append(features.cpu())
    all_labels.extend(labels)
