import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from torchvision.datasets import PCAM
from torch.utils.data import Subset
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extraction import extract_features
from utils.prediction_head import train_prediction_head
from PLS.loss_pls import LossPLS
from PLS.confidence_pls import ConfidencePLS
from injector.sidecar import SidecarModel, InjectorTrainer


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


def compute_similarity_matrix(loss_pls, conf_pls):
    """Computes cosine similarity between loss and confidence components."""
    loss_comps = loss_pls.get_components()
    conf_comps = conf_pls.get_components()
    
    # Normalize
    loss_comps_norm = loss_comps / np.linalg.norm(loss_comps, axis=0, keepdims=True)
    conf_comps_norm = conf_comps / np.linalg.norm(conf_comps, axis=0, keepdims=True)
    
    # Cosine similarity
    similarity = np.dot(loss_comps_norm.T, conf_comps_norm)
    return similarity


def print_similarity_matrix(similarity):
    """Pretty prints the similarity matrix."""
    n_loss, n_conf = similarity.shape
    
    print("\n" + "="*60)
    print("Cosine Similarity Matrix (Loss vs Confidence Components)")
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


def get_user_selection(similarity):
    """Prompts user to select loss and confidence component indices."""
    print("\nPositive correlation = High Loss aligns with High Confidence (Confidently Wrong)")
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


def compute_spurious_direction(loss_pls, conf_pls, loss_idx, conf_idx):
    """
    Computes the spurious direction v as the sum of normalized components.
    """
    loss_comps = loss_pls.get_components()
    conf_comps = conf_pls.get_components()
    
    # Get selected components
    loss_vec = loss_comps[:, loss_idx]
    conf_vec = conf_comps[:, conf_idx]
    
    # Normalize
    loss_vec = loss_vec / np.linalg.norm(loss_vec)
    conf_vec = conf_vec / np.linalg.norm(conf_vec)
    
    # Sum and normalize
    v = loss_vec + conf_vec
    v = v / np.linalg.norm(v)
    
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


def main():
    # Hyperparameters
    num_samples = 10000  # Number of samples for training
    n_components = 5
    epochs = 5
    batch_size = 64
    lr = 1e-4
    lambda_val = 1.0  # Scaling for spurious direction
    alpha_penalty = 1.0  # Weight for invariance penalty
    
    print("="*60)
    print("  SIDECAR INJECTOR TRAINING")
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

    print("\n--- 2. Loading PatchCamelyon Dataset ---")
    try:
        data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_root, exist_ok=True)
        
        full_dataset = PCAM(root=data_root, split='train', download=True)
        
        # Training subset
        indices = torch.arange(num_samples)
        dataset = Subset(full_dataset, indices)
        
        # Validation subset
        val_samples = 10000
        val_indices = torch.arange(num_samples, num_samples + val_samples)
        val_dataset = Subset(full_dataset, val_indices)
        
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
        hidden_dims=[256, 128],
        epochs=10,
        batch_size=64,
        val_features=val_features,
        val_labels=val_labels
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

    # --- Step 2: Global PLS Analysis ---
    print("\n--- 5. Computing Global PLS Components ---")
    loss_pls = LossPLS(features, labels, prediction_head)
    loss_pls.fit(n_components=n_components)
    
    conf_pls = ConfidencePLS(features, labels, prediction_head)
    conf_pls.fit(n_components=n_components)

    # --- Step 3: Interactive Selection ---
    similarity = compute_similarity_matrix(loss_pls, conf_pls)
    print_similarity_matrix(similarity)
    
    loss_idx, conf_idx = get_user_selection(similarity)
    
    v = compute_spurious_direction(loss_pls, conf_pls, loss_idx, conf_idx)
    print(f"\nSpurious direction v computed (shape: {v.shape}).")

    # --- Baseline Sensitivity ---
    print("\n--- Baseline Sensitivity to Direction v ---")
    baseline_sensitivity = compute_sensitivity(prediction_head, val_features, v, lambda_val=lambda_val)
    print(f"Baseline Sensitivity (avg logit change): {baseline_sensitivity:.4f}")

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
    main()

