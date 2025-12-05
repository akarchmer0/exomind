import sys
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
from torch.utils.data import Subset
import numpy as np
from sklearn.linear_model import LinearRegression

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extraction import extract_features
from utils.prediction_head import train_prediction_head
from PLS.loss_pls import LossPLS
from PLS.confidence_pls import ConfidencePLS
from PLS.combined_pls import CombinedPLS
from injector.sidecar import SidecarModel, InjectorTrainer
from spurious_utils import SpuriousPCAM, RotationSpuriousPCAM


def convert_chexpert_to_multiclass(dataset):
    """
    Converts CheXpert multilabel dataset to multiclass.
    CheXpert has 14 pathologies. We convert to multiclass by:
    - If "No Finding" is positive -> class 0
    - Otherwise, assign to first positive pathology (classes 1-13)
    - If no positive labels, assign to class 0 (No Finding)
    
    Returns a new dataset with 'image' and 'label' keys.
    """
    # CheXpert pathology names (in order) - try common variations
    pathology_names = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    # Also try lowercase and snake_case variations
    pathology_variations = []
    for name in pathology_names:
        variations = [
            name,
            name.lower(),
            name.replace(' ', '_').lower(),
            name.replace(' ', '_')
        ]
        pathology_variations.append(variations)
    
    def convert_labels(example):
        # Get all label columns - try to find pathology columns
        labels = []
        no_finding_idx = 0
        
        # Check which columns exist in the dataset
        available_cols = set(example.keys())
        
        # Try to match pathology names to columns
        for i, path_variations in enumerate(pathology_variations):
            found_col = None
            for var in path_variations:
                if var in available_cols:
                    found_col = var
                    break
            
            if found_col:
                val = example[found_col]
                # Handle NaN, -1 (uncertain), 0 (negative), 1 (positive)
                # Convert to float first to handle NaN
                try:
                    val_float = float(val) if val is not None else 0.0
                    if val_float == 1.0:  # Positive
                        labels.append(1)
                    else:
                        labels.append(0)
                except (ValueError, TypeError):
                    labels.append(0)
            else:
                labels.append(0)
        
        # Convert to multiclass: No Finding (class 0) vs first pathology (classes 1-13)
        if labels[no_finding_idx] == 1:  # No Finding is positive
            class_label = 0
        else:
            # Find first positive pathology (skip index 0 which is No Finding)
            found = False
            for i in range(1, len(labels)):
                if labels[i] == 1:
                    class_label = i
                    found = True
                    break
            if not found:
                class_label = 0  # Default to No Finding if no positive labels
        
        return {'image': example['image'], 'label': class_label}
    
    # Apply conversion - keep only image and label columns
    cols_to_remove = [col for col in dataset.column_names if col != 'image']
    converted = dataset.map(convert_labels, remove_columns=cols_to_remove)
    return converted


class InjectorDataset(Dataset):
    """
    Dataset that provides (image_tensor, phikon_embedding, label) tuples.
    Pre-computes Phikon embeddings to avoid running Phikon during training.
    """
    def __init__(self, base_dataset, features, labels, processor):
        """
        Args:
            base_dataset: The original dataset (e.g., PCAM Subset).
            features: Pre-computed Phikon embeddings (tensor).
            labels: Labels (tensor).
            processor: HuggingFace image processor for preprocessing.
        """
        self.base_dataset = base_dataset
        self.features = features
        self.labels = labels
        self.processor = processor
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get image from base dataset
        sample = self.base_dataset[idx]
        if isinstance(sample, (tuple, list)):
            img = sample[0]
        elif isinstance(sample, dict):
            img = sample['image']
        else:
            img = sample
            
        # Process image for ResNet (sidecar)
        # ResNet expects (C, H, W) tensor, normalized
        # Using standard ImageNet normalization
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img)
        
        return img_tensor, self.features[idx], self.labels[idx]


def compute_variance_explained(projections, target):
    """
    Computes R² (variance explained) for a component.
    
    Args:
        projections: 1D array of projections onto component
        target: 1D array of target values (loss or confidence)
    
    Returns:
        r_squared: Variance explained (0-1)
    """
    # Fit linear regression: target ~ projections
    # R² = 1 - (SS_res / SS_tot)
    projections = projections.reshape(-1, 1)
    target = target.reshape(-1, 1)
    
    # Compute mean of target
    target_mean = np.mean(target)
    
    # Fit linear model
    reg = LinearRegression()
    reg.fit(projections, target)
    target_pred = reg.predict(projections)
    
    # Compute R²
    ss_res = np.sum((target - target_pred) ** 2)
    ss_tot = np.sum((target - target_mean) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def align_component_signs(loss_pls, conf_pls, features, labels, prediction_head):
    """
    Aligns the signs of PLS components so that positive projection = increasing target.
    For Loss components: positive projection should mean HIGHER loss.
    For Confidence components: positive projection should mean HIGHER confidence (lower entropy).
    
    Returns:
        loss_signs: Array of +1 or -1 for each loss component.
        conf_signs: Array of +1 or -1 for each confidence component.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_dev = features.to(device)
    labels_dev = labels.to(device)
    prediction_head.eval()
    
    loss_comps = loss_pls.get_components()
    conf_comps = conf_pls.get_components()
    
    n_loss = loss_comps.shape[1]
    n_conf = conf_comps.shape[1]
    
    with torch.no_grad():
        # Compute actual losses
        logits = prediction_head(features_dev)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        losses = criterion(logits, labels_dev).cpu().numpy()
        
        # Compute confidence (negative entropy)
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        confidence = -entropy.cpu().numpy()  # Higher = more confident
    
    features_np = features.cpu().numpy()
    
    # Determine sign for each Loss component
    loss_signs = np.ones(n_loss)
    print("\nAligning Loss component signs:")
    for i in range(n_loss):
        vec = loss_comps[:, i]
        vec = vec / np.linalg.norm(vec)
        projections = features_np @ vec
        corr = np.corrcoef(projections, losses)[0, 1]
        
        # Compute variance explained
        var_explained = compute_variance_explained(projections, losses)
        
        if corr < 0:
            loss_signs[i] = -1
            print(f"  L{i+1}: corr={corr:+.4f}, var_explained={var_explained:.4f} ({var_explained*100:.2f}%) -> FLIP")
        else:
            print(f"  L{i+1}: corr={corr:+.4f}, var_explained={var_explained:.4f} ({var_explained*100:.2f}%) -> OK")
    
    # Determine sign for each Confidence component
    conf_signs = np.ones(n_conf)
    print("\nAligning Confidence component signs:")
    for i in range(n_conf):
        vec = conf_comps[:, i]
        vec = vec / np.linalg.norm(vec)
        projections = features_np @ vec
        corr = np.corrcoef(projections, confidence)[0, 1]
        
        # Compute variance explained
        var_explained = compute_variance_explained(projections, confidence)
        
        if corr < 0:
            conf_signs[i] = -1
            print(f"  C{i+1}: corr={corr:+.4f}, var_explained={var_explained:.4f} ({var_explained*100:.2f}%) -> FLIP")
        else:
            print(f"  C{i+1}: corr={corr:+.4f}, var_explained={var_explained:.4f} ({var_explained*100:.2f}%) -> OK")
    
    return loss_signs, conf_signs


def compute_similarity_matrix(loss_pls, conf_pls, loss_signs=None, conf_signs=None):
    """Computes cosine similarity between loss and confidence components."""
    loss_comps = loss_pls.get_components()
    conf_comps = conf_pls.get_components()
    
    # Apply signs if provided
    if loss_signs is not None:
        loss_comps = loss_comps * loss_signs
    if conf_signs is not None:
        conf_comps = conf_comps * conf_signs
    
    # Normalize
    loss_comps_norm = loss_comps / np.linalg.norm(loss_comps, axis=0, keepdims=True)
    conf_comps_norm = conf_comps / np.linalg.norm(conf_comps, axis=0, keepdims=True)
    
    # Cosine similarity
    similarity = np.dot(loss_comps_norm.T, conf_comps_norm)
    return similarity


def print_similarity_matrix(similarity, class_label=None):
    """Pretty prints the similarity matrix."""
    n_loss, n_conf = similarity.shape
    
    class_str = f" - Class {class_label}" if class_label is not None else ""
    print("\n" + "="*60)
    print(f"Cosine Similarity Matrix (Loss vs Confidence Components){class_str}")
    print("="*60)
    
    # Header
    header = "       " + "".join([f"  C{j+1}   " for j in range(n_conf)])
    print(header)
    print("-" * len(header))
    
    for i in range(n_loss):
        row = f"  L{i+1}  "
        for j in range(n_conf):
            row += f" {similarity[i, j]:+.3f} "
        print(row)
    
    print("="*60)


def get_user_selection(similarity, class_label=None):
    """Prompts user to select loss and confidence component indices."""
    class_str = f" (Class {class_label})" if class_label is not None else ""
    print(f"\nPositive correlation = High Loss aligns with High Confidence (Confidently Wrong){class_str}")
    print("Select the pair representing the spurious direction to remove.\n")
    
    while True:
        try:
            loss_idx = int(input("Enter Loss component index (1-based, e.g., 1 for L1): ")) - 1
            conf_idx = int(input("Enter Confidence component index (1-based, e.g., 1 for C1): ")) - 1
            
            if 0 <= loss_idx < similarity.shape[0] and 0 <= conf_idx < similarity.shape[1]:
                print(f"\nSelected: L{loss_idx+1} and C{conf_idx+1} (Similarity: {similarity[loss_idx, conf_idx]:.3f})")
                return loss_idx, conf_idx
            else:
                print("Invalid indices. Try again.")
        except ValueError:
            print("Please enter valid integers.")


def compute_spurious_direction(loss_pls, conf_pls, loss_idx, conf_idx, loss_signs, conf_signs):
    """
    Computes the spurious direction v as the sum of sign-corrected normalized components.
    """
    loss_comps = loss_pls.get_components()
    conf_comps = conf_pls.get_components()
    
    # Get selected components and apply the correct sign
    loss_vec = loss_comps[:, loss_idx] * loss_signs[loss_idx]
    conf_vec = conf_comps[:, conf_idx] * conf_signs[conf_idx]
    
    # Normalize
    loss_vec = loss_vec / np.linalg.norm(loss_vec)
    conf_vec = conf_vec / np.linalg.norm(conf_vec)
    
    # Compute spurious direction v as the sum of the selected components
    v = loss_vec + conf_vec
    # Normalize
    v = v / np.linalg.norm(v)
    
    print(f"\nSpurious direction v:")
    print(f"  L{loss_idx+1} sign: {'+' if loss_signs[loss_idx] > 0 else '-'}")
    print(f"  C{conf_idx+1} sign: {'+' if conf_signs[conf_idx] > 0 else '-'}")
    
    return torch.tensor(v, dtype=torch.float32)


def compute_sensitivity(prediction_head, features, v, lambda_val=1.0, device=None):
    """
    Computes the sensitivity of the prediction head to the direction v.
    Sensitivity = average L2 norm of (logits - logits_perturbed) when perturbing by lambda * v.
    
    Args:
        prediction_head: The prediction head model.
        features: Tensor of embeddings (N, D).
        v: The direction tensor (D,).
        lambda_val: Scaling factor for perturbation.
        device: Device to use.
        
    Returns:
        avg_sensitivity: Average L2 distance between original and perturbed logits.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prediction_head.eval()
    features = features.to(device)
    v = v.to(device)
    
    with torch.no_grad():
        logits_orig = prediction_head(features)
        features_perturbed = features - lambda_val * v.unsqueeze(0)
        logits_perturbed = prediction_head(features_perturbed)
        
        # L2 distance per sample
        diff = logits_orig - logits_perturbed
        l2_per_sample = torch.norm(diff, dim=1)
        avg_sensitivity = l2_per_sample.mean().item()
    
    return avg_sensitivity


def compute_sensitivity_with_sidecar(trainer, val_loader, v, lambda_val=1.0):
    """
    Computes the sensitivity of the combined model (Phikon + Sidecar + Head) to direction v.
    
    Args:
        trainer: InjectorTrainer instance.
        val_loader: DataLoader yielding (images, h_phikon, labels).
        v: The direction tensor (D,).
        lambda_val: Scaling factor for perturbation.
        
    Returns:
        avg_sensitivity: Average L2 distance between original and perturbed logits.
    """
    trainer.sidecar.eval()
    v = v.to(trainer.device)
    
    total_sensitivity = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images, h_phikon, labels = batch
            images = images.to(trainer.device)
            h_phikon = h_phikon.to(trainer.device)
            
            h_sidecar = trainer.sidecar(images)
            h_combined = h_phikon + h_sidecar
            
            logits_orig = trainer.prediction_head(h_combined)
            
            h_perturbed = h_combined - lambda_val * v.unsqueeze(0)
            logits_perturbed = trainer.prediction_head(h_perturbed)
            
            diff = logits_orig - logits_perturbed
            l2_per_sample = torch.norm(diff, dim=1)
            
            total_sensitivity += l2_per_sample.sum().item()
            total_samples += labels.size(0)
    
    return total_sensitivity / total_samples if total_samples > 0 else 0


def evaluate_sidecar_val_loss(trainer, val_loader):
    """Evaluates sidecar on validation set and returns loss."""
    trainer.sidecar.eval()
    total_val_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images, h_phikon, batch_labels = batch
            images = images.to(trainer.device)
            h_phikon = h_phikon.to(trainer.device)
            batch_labels = batch_labels.to(trainer.device)
            
            h_sidecar = trainer.sidecar(images)
            h_combined = h_phikon + h_sidecar
            logits = trainer.prediction_head(h_combined)
            
            loss = trainer.ce_loss(logits, batch_labels)
            total_val_loss += loss.item() * batch_labels.size(0)
            total_samples += batch_labels.size(0)
            
    return total_val_loss / total_samples if total_samples > 0 else 0


def get_top_spurious_indices(features, v, percentile=1):
    """
    Returns indices of samples that score in the top percentile on the spurious direction v.
    
    Args:
        features: Tensor of shape (N, D).
        v: Direction tensor of shape (D,).
        percentile: Top percentile to select (default 1 = top 1%).
        
    Returns:
        indices: Tensor of indices for top scoring samples.
    """
    # Project features onto v
    v_normalized = v / torch.norm(v)
    projections = torch.matmul(features, v_normalized)
    
    # Find threshold for top percentile
    k = max(1, int(len(projections) * percentile / 100))
    threshold = torch.topk(projections, k).values[-1]
    
    # Get indices above threshold
    indices = (projections >= threshold).nonzero(as_tuple=True)[0]
    
    return indices


def main(args):
    # Hyperparameters
    num_samples = 10000  # Number of samples for training
    n_components = 5
    epochs = 5
    batch_size = 64
    lr = 1e-4
    lambda_val = 1.0  # Scaling for spurious direction
    alpha_penalty = 0.5  # Weight for invariance penalty
    
    print("="*60)
    print("  SIDECAR INJECTOR TRAINING")
    print(f"  Dataset: {args.dataset}")
    print("="*60)
    
    # --- Step 1: Setup ---
    print("\n--- 1. Loading Phikon Model and Processor ---")
    try:
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        phikon_model = AutoModel.from_pretrained("owkin/phikon")
        print("Phikon loaded successfully.")
    except Exception as e:
        print(f"Error loading Phikon: {e}")
        return

    print("\n--- 2. Loading Dataset ---")
    try:
        if args.dataset == 'chexpert':
            # Load CheXpert from HuggingFace
            print("Loading CheXpert from HuggingFace datasets...")
            try:
                # Load CheXpert dataset
                full_dataset = load_dataset("danjacobellis/chexpert", split='train')
                print(f"Loaded {len(full_dataset)} samples from CheXpert.")
                
                # Convert multilabel to multiclass
                print("Converting multilabel to multiclass...")
                full_dataset = convert_chexpert_to_multiclass(full_dataset)
                
                # CheXpert is large, use reasonable sample sizes
                chexpert_num_samples = min(100000, len(full_dataset))  # Use up to 50k samples
                chexpert_val_samples = min(20000, len(full_dataset) - chexpert_num_samples)
                
                indices = list(range(chexpert_num_samples))
                train_subset = full_dataset.select(indices)
                
                val_indices = list(range(chexpert_num_samples, chexpert_num_samples + chexpert_val_samples))
                val_subset = full_dataset.select(val_indices)
                
                dataset = train_subset
                val_dataset = val_subset
                print(f"Using CheXpert dataset (14 classes: No Finding + 13 pathologies).")
                print(f"  Classes: 0=No Finding, 1-13=Pathologies (Enlarged Cardiomediastinum, Cardiomegaly, etc.)")
            except Exception as e:
                print(f"Error loading CheXpert dataset: {e}")
                print("Note: CheXpert dataset may require additional setup.")
                raise
        elif args.dataset == 'kather':
            # Load Kather100K (NCT-CRC-HE-100K) from HuggingFace
            print("Loading Kather100K (NCT-CRC-HE-100K) from HuggingFace datasets...")
            try:
                # Try loading from HuggingFace - common dataset names
                # Option 1: Direct HuggingFace dataset if available
                try:
                    full_dataset = load_dataset("timm/NCT-CRC-HE-100K", split='train')
                except:
                    # Option 2: ImageFolder format (requires local path)
                    data_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'NCT-CRC-HE-100K')
                    if os.path.exists(data_root):
                        full_dataset = load_dataset("imagefolder", data_dir=data_root, split='train')
                    else:
                        raise ValueError(f"Kather dataset not found at {data_root}. Please download NCT-CRC-HE-100K dataset.")
            except Exception as e:
                print(f"Error loading Kather dataset: {e}")
                print("Note: Kather dataset may need to be downloaded separately.")
                print("Please download NCT-CRC-HE-100K and place it in data/NCT-CRC-HE-100K/")
                raise
            
            # Kather has 9 classes, use more samples for multiclass
            kather_num_samples = min(50000, len(full_dataset))  # Use up to 50k samples
            kather_val_samples = min(10000, len(full_dataset) - kather_num_samples)
            
            indices = list(range(kather_num_samples))
            train_subset = full_dataset.select(indices)
            
            val_indices = list(range(kather_num_samples, kather_num_samples + kather_val_samples))
            val_subset = full_dataset.select(val_indices)
            
            dataset = train_subset
            val_dataset = val_subset
            print(f"Using Kather100K dataset (9 tissue classes).")
            print(f"  Classes: 0-8 (Adipose, Background, Debris, Lymphocytes, Mucus, Smooth muscle, Normal colon, Cancer-associated stroma, Cancer epithelium)")
        else:
            # Load PCAM from HuggingFace (avoids Google Drive rate limits)
            print("Loading PCAM from HuggingFace datasets...")
            full_dataset = load_dataset("1aurent/PatchCamelyon", split='train')
            
            # Training subset
            indices = list(range(num_samples))
            train_subset = full_dataset.select(indices)
            
            # Validation subset
            val_samples = 2000
            val_indices = list(range(num_samples, num_samples + val_samples))
            val_subset = full_dataset.select(val_indices)
            
            # Wrap with spurious correlation dataset if specified
            # Note: HuggingFace datasets return dicts with 'image' and 'label' keys
            if args.dataset == 'normal':
                dataset = train_subset
                val_dataset = val_subset
                print("Using normal (unmodified) PCAM dataset.")
            elif args.dataset == 'spurious':
                artifact_probs = {0: 0.0, 1: 0.9}
                # Training: Black square only on positive class (spurious correlation)
                dataset = SpuriousPCAM(train_subset, artifact_probs=artifact_probs, square_size=30)
                # Validation: Black square on both classes (balanced, no correlation)
                val_dataset = SpuriousPCAM(val_subset, artifact_probs=artifact_probs, square_size=30)
                print("Using SpuriousPCAM (black square artifact).")
                print(f"  Training: artifact on {artifact_probs[1]*100}% of positive class only.")
                print(f"  Validation: artifact on {artifact_probs[0]*100}% of both classes (balanced).")
            elif args.dataset == 'rotation':
                artifact_probs = {0: 0.0, 1: 0.9}
                # Training: Rotation only on positive class (spurious correlation)  
                dataset = RotationSpuriousPCAM(train_subset, artifact_probs=artifact_probs)
                # Validation: Rotation on both classes (balanced, no correlation)
                val_dataset = RotationSpuriousPCAM(val_subset, artifact_probs=artifact_probs)
                print("Using RotationSpuriousPCAM (90-degree rotation artifact).")
                print(f"  Training: artifact on {artifact_probs[1]*100}% of positive class only.")
                print(f"  Validation: artifact on {artifact_probs[0]*100}% of both classes (balanced).")
        
        print(f"Training samples: {len(dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\n--- 3. Extracting Features (Global) ---")
    print("Extracting training features...")
    features, labels = extract_features(phikon_model, processor, dataset)
    print(f"Train Features shape: {features.shape}")
    
    print("Extracting validation features...")
    val_features, val_labels = extract_features(phikon_model, processor, val_dataset)
    print(f"Val Features shape: {val_features.shape}")

    print("\n--- 4. Training Prediction Head ---")
    prediction_head = train_prediction_head(
        features=features,
        labels=labels,
        hidden_dims=[512, 256, 128],
        epochs=20,
        batch_size=64,
        val_features=val_features,
        val_labels=val_labels,
        dropout_rate=0.2,
        weight_decay=1e-5
    )
    print("Prediction head trained (best val loss saved).")
    
    # Evaluate baseline performance on validation set
    print("\n--- Baseline Evaluation on Validation Set ---")
    prediction_head.eval()
    val_features_dev = val_features.to("cuda" if torch.cuda.is_available() else "cpu")
    val_labels_dev = val_labels.to("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = prediction_head(val_features_dev)
        base_loss = criterion(logits, val_labels_dev).item()
        _, predicted = torch.max(logits, 1)
        base_acc = (predicted == val_labels_dev).sum().item() / len(val_labels_dev)
    
    print(f"Baseline Val Loss: {base_loss:.4f}")
    print(f"Baseline Val Acc:  {base_acc*100:.2f}%")

    # --- Step 2: Per-Class PLS Analysis (on Validation/Analysis Set) ---
    print("\n--- 5. Computing Per-Class PLS Components (on Analysis Set) ---")
    print("Note: PLS runs on validation set (balanced artifact) to detect model bias.")
    
    unique_classes = torch.unique(val_labels).tolist()
    pls_data = {}  # Store PLS models and signs for each class
    
    for c in unique_classes:
        print(f"\n{'='*40}")
        print(f"  CLASS {c}")
        print(f"{'='*40}")
        
        # Filter data for this class (using validation/analysis set)
        class_mask = (val_labels == c)
        features_c = val_features[class_mask]
        labels_c = val_labels[class_mask]
        
        print(f"Samples in Class {c}: {len(features_c)}")
        
        # Compute PLS for this class
        loss_pls_c = LossPLS(features_c, labels_c, prediction_head)
        loss_pls_c.fit(n_components=n_components)
        
        conf_pls_c = ConfidencePLS(features_c, labels_c, prediction_head)
        conf_pls_c.fit(n_components=n_components)
        
        # Compute Combined PLS (multivariate on [loss, confidence])
        combined_pls_c = CombinedPLS(features_c, labels_c, prediction_head)
        combined_pls_c.fit(n_components=n_components)
        combined_pls_c.print_component_stats(n_components=n_components)
        
        # Align component signs (for individual PLS visualization)
        print(f"\nAligning signs for Class {c}:")
        loss_signs_c, conf_signs_c = align_component_signs(loss_pls_c, conf_pls_c, features_c, labels_c, prediction_head)
        
        # Compute similarity matrix
        similarity_c = compute_similarity_matrix(loss_pls_c, conf_pls_c, loss_signs_c, conf_signs_c)
        
        # Store for later use
        pls_data[c] = {
            'loss_pls': loss_pls_c,
            'conf_pls': conf_pls_c,
            'combined_pls': combined_pls_c,
            'loss_signs': loss_signs_c,
            'conf_signs': conf_signs_c,
            'similarity': similarity_c
        }

    # --- Step 3: Display All Similarity Matrices ---
    print("\n" + "="*60)
    print("  ALL SIMILARITY MATRICES")
    print("="*60)
    
    for c in unique_classes:
        print_similarity_matrix(pls_data[c]['similarity'], class_label=c)
    
    # --- Step 4: Interactive Selection ---
    print("\n" + "="*60)
    print("  SELECT SPURIOUS DIRECTION (via Combined PLS)")
    print("="*60)
    
    while True:
        try:
            selected_class = int(input(f"\nSelect class to use for spurious direction ({unique_classes}): "))
            if selected_class in unique_classes:
                break
            else:
                print(f"Invalid class. Choose from {unique_classes}")
        except ValueError:
            print("Please enter a valid integer.")
    
    print(f"\nUsing Class {selected_class} Combined PLS first component.")
    
    # Show similarity matrix for reference
    print_similarity_matrix(pls_data[selected_class]['similarity'], class_label=selected_class)
    
    # Get spurious direction from Combined PLS (first component)
    combined_pls = pls_data[selected_class]['combined_pls']
    combined_pls.print_component_stats(n_components=3)
    
    # Determine sign and get direction
    sign = combined_pls.align_sign()
    v = combined_pls.get_spurious_direction(component_idx=0) * sign
    
    var_explained = combined_pls.get_variance_explained()
    print(f"\nSpurious direction v (Combined PLS Component 1):")
    print(f"  Sign applied: {'+' if sign > 0 else '-'}")
    print(f"  Loss R²: {var_explained['loss'][0]:.4f} ({var_explained['loss'][0]*100:.2f}%)")
    print(f"  Confidence R²: {var_explained['confidence'][0]:.4f} ({var_explained['confidence'][0]*100:.2f}%)")
    print(f"  Shape: {v.shape}")

    # --- Baseline Sensitivity ---
    print("\n--- Baseline Sensitivity to Direction v ---")
    baseline_sensitivity = compute_sensitivity(prediction_head, val_features, v, lambda_val=lambda_val)
    print(f"Baseline Sensitivity (avg logit change): {baseline_sensitivity:.4f}")

    # =========================================================================
    # LINEAR CONCEPT ERASURE (If requested)
    # =========================================================================
    if args.use_linear_erasure:
        print("\n" + "="*60)
        print("  APPLYING LINEAR CONCEPT ERASURE")
        print("="*60)
        
        # Project out the spurious direction v from all features
        # h_clean = h - (h . v) * v
        # Ensure v is normalized
        v_norm = v / torch.norm(v)
        v_norm = v_norm.to(features.device)
        
        # 1. Clean Training Features
        print("Projecting out v from Training Features...")
        # features: (N, D), v: (D) -> dot product per sample
        projections_train = torch.matmul(features, v_norm) # (N,)
        features_clean = features - projections_train.unsqueeze(1) * v_norm.unsqueeze(0)
        
        # 2. Clean Validation Features
        print("Projecting out v from Validation Features...")
        projections_val = torch.matmul(val_features, v_norm)
        val_features_clean = val_features - projections_val.unsqueeze(1) * v_norm.unsqueeze(0)
        
        # 3. Retrain Prediction Head on Clean Features
        print("\n--- Retraining Prediction Head on CLEANED Features ---")
        prediction_head_clean = train_prediction_head(
            features=features_clean,
            labels=labels,
            hidden_dims=[512, 256, 128],
            epochs=epochs,
            batch_size=batch_size,
            val_features=val_features_clean,
            val_labels=val_labels,
            dropout_rate=0.2,
            weight_decay=1e-5
        )
        
        # 4. Evaluate Clean Head
        print("\n--- Evaluation of Linear Erasure ---")
        prediction_head_clean.eval()
        val_features_clean_dev = val_features_clean.to("cuda" if torch.cuda.is_available() else "cpu")
        val_labels_dev = val_labels.to("cuda" if torch.cuda.is_available() else "cpu")
        
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = prediction_head_clean(val_features_clean_dev)
            clean_loss = criterion(logits, val_labels_dev).item()
            _, predicted = torch.max(logits, 1)
            clean_acc = (predicted == val_labels_dev).sum().item() / len(val_labels_dev)
            
        print(f"Linear Erasure Val Loss: {clean_loss:.4f}")
        print(f"Linear Erasure Val Acc:  {clean_acc*100:.2f}%")
        print(f"Change vs Baseline Acc:  {clean_acc*100 - base_acc*100:+.2f}%")
        
        # Sensitivity of Clean Head to v (should be 0)
        # Note: perturbing cleaned features by v has no effect if v is orthogonal, 
        # BUT here compute_sensitivity adds v to the input. 
        # The clean head expects inputs orthogonal to v. 
        # If we add v, and the first layer is linear, it might still react unless we explicitly project input in the model.
        # However, if we retrained on data orthogonal to v, the weights W corresponding to v (W . v) should be 0 or random noise.
        
        clean_sensitivity = compute_sensitivity(prediction_head_clean, val_features_clean, v, lambda_val=lambda_val)
        print(f"Sensitivity of Clean Head (retrained) to v: {clean_sensitivity:.4f}")
        
        # =========================================================================
        # TOP 1% SPURIOUS SUBSET EVALUATION (Linear Erasure)
        # =========================================================================
        print("\n" + "="*60)
        print("  TOP 1% SPURIOUS SUBSET EVALUATION")
        print("="*60)
        
        # Get top 1% indices based on projection onto v (Combined PLS first component)
        top_indices = get_top_spurious_indices(val_features, v, percentile=1)
        print(f"Evaluating on {len(top_indices)} samples (top 1% on Combined PLS component)")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Baseline on top 1%
        top_features = val_features[top_indices].to(device)
        top_labels = val_labels[top_indices].to(device)
        with torch.no_grad():
            logits = prediction_head(top_features)
            top_base_loss = criterion(logits, top_labels).item()
            _, predicted = torch.max(logits, 1)
            top_base_acc = (predicted == top_labels).sum().item() / len(top_labels)
        
        print(f"Baseline - Top 1% Loss: {top_base_loss:.4f}, Acc: {top_base_acc*100:.2f}%")
        
        # Linear Erasure on top 1%
        top_features_clean = val_features_clean[top_indices].to(device)
        with torch.no_grad():
            logits = prediction_head_clean(top_features_clean)
            top_clean_loss = criterion(logits, top_labels).item()
            _, predicted = torch.max(logits, 1)
            top_clean_acc = (predicted == top_labels).sum().item() / len(top_labels)
        
        print(f"Linear Erasure - Top 1% Loss: {top_clean_loss:.4f}, Acc: {top_clean_acc*100:.2f}%")
        print(f"Change in Top 1% Acc: {top_clean_acc*100 - top_base_acc*100:+.2f}%")
        
        # Summary
        print("\n" + "="*60)
        print("  SUMMARY (Linear Erasure)")
        print("="*60)
        print(f"{'Metric':<30} {'Baseline':<15} {'Linear Erasure':<15} {'Change':<15}")
        print("-"*75)
        print(f"{'Overall Val Loss':<30} {base_loss:<15.4f} {clean_loss:<15.4f} {clean_loss - base_loss:+.4f}")
        print(f"{'Overall Val Acc (%)':<30} {base_acc*100:<15.2f} {clean_acc*100:<15.2f} {clean_acc*100 - base_acc*100:+.2f}")
        print(f"{'Top 1% Val Loss':<30} {top_base_loss:<15.4f} {top_clean_loss:<15.4f} {top_clean_loss - top_base_loss:+.4f}")
        print(f"{'Top 1% Val Acc (%)':<30} {top_base_acc*100:<15.2f} {top_clean_acc*100:<15.2f} {top_clean_acc*100 - top_base_acc*100:+.2f}")
        print(f"{'Sensitivity to v':<30} {baseline_sensitivity:<15.4f} {clean_sensitivity:<15.4f} {clean_sensitivity - baseline_sensitivity:+.4f}")
        print("="*60)
        
        return # Exit after linear erasure experiment

    # Prepare validation loader (shared by both sidecar versions)
    val_injector_dataset = InjectorDataset(val_dataset, val_features, val_labels, processor)
    val_loader = DataLoader(val_injector_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # =========================================================================
    # TRAIN SIDECAR WITHOUT INVARIANCE PENALTY (alpha=0)
    # =========================================================================
    print("\n" + "="*60)
    print("  TRAINING SIDECAR WITHOUT INVARIANCE PENALTY (alpha=0)")
    print("="*60)
    
    sidecar_no_penalty = SidecarModel(output_dim=768, pretrained=False)
    trainer_no_penalty = InjectorTrainer(prediction_head, sidecar_no_penalty, v)
    optimizer_no_penalty = optim.Adam(sidecar_no_penalty.parameters(), lr=lr)
    
    injector_dataset = InjectorDataset(dataset, features, labels, processor)
    train_loader = DataLoader(injector_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"Training for {epochs} epochs (NO invariance penalty)...\n")
    
    best_val_loss_no_penalty = float('inf')
    best_state_no_penalty = None
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        avg_loss, avg_ce, avg_penalty = trainer_no_penalty.train_epoch(
            train_loader, optimizer_no_penalty, lambda_val=lambda_val, alpha_penalty=0.0
        )
        train_acc = trainer_no_penalty.evaluate(train_loader)
        
        # Validation
        val_loss_epoch = evaluate_sidecar_val_loss(trainer_no_penalty, val_loader)
        val_acc_epoch = trainer_no_penalty.evaluate(val_loader)
        
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch*100:.2f}%")
        
        if val_loss_epoch < best_val_loss_no_penalty:
            best_val_loss_no_penalty = val_loss_epoch
            best_state_no_penalty = sidecar_no_penalty.state_dict().copy()
            print(f"  [New Best Model Saved]")
        print()
    
    # Load best model
    if best_state_no_penalty is not None:
        sidecar_no_penalty.load_state_dict(best_state_no_penalty)
        print(f"Loaded best No-Penalty model with Val Loss: {best_val_loss_no_penalty:.4f}")
    
    # Evaluate on validation set
    print("\n--- Evaluation (NO Invariance Penalty) on Validation Set ---")
    no_penalty_acc = trainer_no_penalty.evaluate(val_loader)
    
    total_val_loss = 0
    total_samples = 0
    trainer_no_penalty.sidecar.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, h_phikon, batch_labels = batch
            images = images.to(trainer_no_penalty.device)
            h_phikon = h_phikon.to(trainer_no_penalty.device)
            batch_labels = batch_labels.to(trainer_no_penalty.device)
            
            h_sidecar = trainer_no_penalty.sidecar(images)
            h_combined = h_phikon + h_sidecar
            logits = trainer_no_penalty.prediction_head(h_combined)
            
            loss = trainer_no_penalty.ce_loss(logits, batch_labels)
            total_val_loss += loss.item() * batch_labels.size(0)
            total_samples += batch_labels.size(0)
            
    no_penalty_loss = total_val_loss / total_samples if total_samples > 0 else 0
    
    print(f"Val Loss (No Penalty): {no_penalty_loss:.4f}")
    print(f"Val Acc (No Penalty):  {no_penalty_acc*100:.2f}%")
    
    no_penalty_sensitivity = compute_sensitivity_with_sidecar(trainer_no_penalty, val_loader, v, lambda_val=lambda_val)
    print(f"Sensitivity (No Penalty): {no_penalty_sensitivity:.4f}")

    # =========================================================================
    # TRAIN SIDECAR WITH INVARIANCE PENALTY
    # =========================================================================
    print("\n" + "="*60)
    print(f"  TRAINING SIDECAR WITH INVARIANCE PENALTY (alpha={alpha_penalty})")
    print("="*60)
    
    sidecar = SidecarModel(output_dim=768, pretrained=False)
    trainer = InjectorTrainer(prediction_head, sidecar, v)
    optimizer = optim.Adam(sidecar.parameters(), lr=lr)
    
    print(f"Lambda: {lambda_val}, Alpha (penalty weight): {alpha_penalty}")
    print(f"Training for {epochs} epochs...\n")
    
    best_val_loss_with_penalty = float('inf')
    best_state_with_penalty = None
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        avg_loss, avg_ce, avg_penalty = trainer.train_epoch(
            train_loader, optimizer, lambda_val=lambda_val, alpha_penalty=alpha_penalty
        )
        train_acc = trainer.evaluate(train_loader)
        
        # Validation
        val_loss_epoch = evaluate_sidecar_val_loss(trainer, val_loader)
        val_acc_epoch = trainer.evaluate(val_loader)
        
        print(f"  Train Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Penalty: {avg_penalty:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch*100:.2f}%")
        
        if val_loss_epoch < best_val_loss_with_penalty:
            best_val_loss_with_penalty = val_loss_epoch
            best_state_with_penalty = sidecar.state_dict().copy()
            print(f"  [New Best Model Saved]")
        print()
    
    # Load best model
    if best_state_with_penalty is not None:
        sidecar.load_state_dict(best_state_with_penalty)
        print(f"Loaded best With-Penalty model with Val Loss: {best_val_loss_with_penalty:.4f}")
    
    print("="*60)
    print("  Training Complete!")
    print("="*60)
    
    # Save the trained sidecar
    save_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, 'sidecar_trained.pth')
    torch.save(sidecar.state_dict(), model_path)
    print(f"Sidecar model saved to {model_path}")

    # --- Final Evaluation on Validation Set ---
    print("\n--- Final Evaluation (WITH Invariance Penalty) on Validation Set ---")
    
    # Use the trainer's evaluate method which handles the combined forward pass
    final_acc = trainer.evaluate(val_loader)
    
    # For loss, we need to compute it manually similar to evaluate but with loss
    total_val_loss = 0
    total_samples = 0
    trainer.sidecar.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, h_phikon, labels = batch
            images = images.to(trainer.device)
            h_phikon = h_phikon.to(trainer.device)
            labels = labels.to(trainer.device)
            
            h_sidecar = trainer.sidecar(images)
            h_combined = h_phikon + h_sidecar
            logits = trainer.prediction_head(h_combined)
            
            loss = trainer.ce_loss(logits, labels)
            total_val_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
    final_loss = total_val_loss / total_samples if total_samples > 0 else 0
    
    print(f"Final Val Loss: {final_loss:.4f}")
    print(f"Final Val Acc:  {final_acc*100:.2f}%")
    
    print(f"Change in Accuracy: {final_acc*100 - base_acc*100:+.2f}%")
    print(f"Change in Loss: {final_loss - base_loss:+.4f}")

    # --- Final Sensitivity ---
    print("\n--- Final Sensitivity to Direction v (with Sidecar) ---")
    final_sensitivity = compute_sensitivity_with_sidecar(trainer, val_loader, v, lambda_val=lambda_val)
    print(f"Final Sensitivity (avg logit change): {final_sensitivity:.4f}")
    print(f"Change in Sensitivity: {final_sensitivity - baseline_sensitivity:+.4f}")

    # =========================================================================
    # TOP 1% SPURIOUS SUBSET EVALUATION
    # =========================================================================
    print("\n" + "="*60)
    print("  TOP 1% SPURIOUS SUBSET EVALUATION")
    print("="*60)
    
    # Get indices of top 1% samples scoring highest on spurious direction v
    top_spurious_indices = get_top_spurious_indices(val_features, v, percentile=1)
    print(f"Evaluating on {len(top_spurious_indices)} samples (top 1% on spurious direction)")
    
    # Create subset dataset and loader for top spurious samples
    top_spurious_features = val_features[top_spurious_indices]
    top_spurious_labels = val_labels[top_spurious_indices]
    # Use .select() for HuggingFace datasets, Subset for others
    if hasattr(val_dataset, 'select'):
        top_spurious_dataset = val_dataset.select(top_spurious_indices.tolist())
    else:
        top_spurious_dataset = Subset(val_dataset, top_spurious_indices.tolist())
    top_spurious_injector_dataset = InjectorDataset(top_spurious_dataset, top_spurious_features, top_spurious_labels, processor)
    top_spurious_loader = DataLoader(top_spurious_injector_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Baseline evaluation on top spurious subset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prediction_head.eval()
    top_spurious_features_dev = top_spurious_features.to(device)
    top_spurious_labels_dev = top_spurious_labels.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = prediction_head(top_spurious_features_dev)
        top_base_loss = criterion(logits, top_spurious_labels_dev).item()
        _, predicted = torch.max(logits, 1)
        top_base_acc = (predicted == top_spurious_labels_dev).sum().item() / len(top_spurious_labels_dev)
    
    print(f"Baseline - Top 1% Loss: {top_base_loss:.4f}, Acc: {top_base_acc*100:.2f}%")
    
    # No Penalty sidecar on top spurious subset
    top_no_penalty_acc = trainer_no_penalty.evaluate(top_spurious_loader)
    top_no_penalty_loss = evaluate_sidecar_val_loss(trainer_no_penalty, top_spurious_loader)
    print(f"No Penalty - Top 1% Loss: {top_no_penalty_loss:.4f}, Acc: {top_no_penalty_acc*100:.2f}%")
    
    # With Penalty sidecar on top spurious subset
    top_final_acc = trainer.evaluate(top_spurious_loader)
    top_final_loss = evaluate_sidecar_val_loss(trainer, top_spurious_loader)
    print(f"With Penalty - Top 1% Loss: {top_final_loss:.4f}, Acc: {top_final_acc*100:.2f}%")

    # =========================================================================
    # SUMMARY COMPARISON
    # =========================================================================
    print("\n" + "="*60)
    print("  SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Metric':<30} {'Baseline':<15} {'No Penalty':<15} {'With Penalty':<15}")
    print("-"*75)
    print(f"{'Val Loss':<30} {base_loss:<15.4f} {no_penalty_loss:<15.4f} {final_loss:<15.4f}")
    print(f"{'Val Accuracy (%)':<30} {base_acc*100:<15.2f} {no_penalty_acc*100:<15.2f} {final_acc*100:<15.2f}")
    print(f"{'Sensitivity to v':<30} {baseline_sensitivity:<15.4f} {no_penalty_sensitivity:<15.4f} {final_sensitivity:<15.4f}")
    print("-"*75)
    print(f"{'Top 1% Val Loss':<30} {top_base_loss:<15.4f} {top_no_penalty_loss:<15.4f} {top_final_loss:<15.4f}")
    print(f"{'Top 1% Val Accuracy (%)':<30} {top_base_acc*100:<15.2f} {top_no_penalty_acc*100:<15.2f} {top_final_acc*100:<15.2f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sidecar Injector for Spurious Correlation Removal")
    parser.add_argument('--dataset', type=str, default='normal',
                        choices=['normal', 'spurious', 'rotation', 'kather', 'chexpert'],
                        help='Dataset variant: normal, spurious (black square), rotation (90deg), kather (multiclass), chexpert (chest xray)')
    parser.add_argument('--use-linear-erasure', action='store_true',
                        help='Apply linear concept erasure (projection) to remove spurious direction')
    args = parser.parse_args()
    main(args)

