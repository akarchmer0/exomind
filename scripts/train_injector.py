import sys
import os
import argparse
import torch
from torch.utils.data import Subset
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
import numpy as np
from sklearn.linear_model import LinearRegression

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extraction import extract_features
from utils.prediction_head import train_prediction_head
from PLS.loss_pls import LossPLS
from PLS.confidence_pls import ConfidencePLS
from PLS.combined_pls import CombinedPLS
from spurious_utils import SpuriousPCAM, RotationSpuriousPCAM


def convert_chexpert_to_multiclass(dataset):
    """
    Converts CheXpert multilabel dataset to multiclass.
    """
    pathology_names = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    pathology_variations = []
    for name in pathology_names:
        variations = [
            name, name.lower(),
            name.replace(' ', '_').lower(),
            name.replace(' ', '_')
        ]
        pathology_variations.append(variations)
    
    def convert_labels(example):
        labels = []
        available_cols = set(example.keys())
        
        for path_variations in pathology_variations:
            found_col = None
            for var in path_variations:
                if var in available_cols:
                    found_col = var
                    break
            
            if found_col:
                val = example[found_col]
                try:
                    val_float = float(val) if val is not None else 0.0
                    labels.append(1 if val_float == 1.0 else 0)
                except (ValueError, TypeError):
                    labels.append(0)
            else:
                labels.append(0)
        
        if labels[0] == 1:
            class_label = 0
        else:
            found = False
            for i in range(1, len(labels)):
                if labels[i] == 1:
                    class_label = i
                    found = True
                    break
            if not found:
                class_label = 0
        
        return {'image': example['image'], 'label': class_label}
    
    cols_to_remove = [col for col in dataset.column_names if col != 'image']
    converted = dataset.map(convert_labels, remove_columns=cols_to_remove)
    return converted


def compute_variance_explained(projections, target):
    """Computes R² (variance explained) for a component."""
    projections = projections.reshape(-1, 1)
    target = target.reshape(-1, 1)
    target_mean = np.mean(target)
    
    reg = LinearRegression()
    reg.fit(projections, target)
    target_pred = reg.predict(projections)
    
    ss_res = np.sum((target - target_pred) ** 2)
    ss_tot = np.sum((target - target_mean) ** 2)
    
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def get_top_spurious_indices(features, v, percentile=1):
    """Returns indices of samples in top percentile on direction v."""
    v_normalized = v / torch.norm(v)
    projections = torch.matmul(features, v_normalized)
    k = max(1, int(len(projections) * percentile / 100))
    threshold = torch.topk(projections, k).values[-1]
    indices = (projections >= threshold).nonzero(as_tuple=True)[0]
    return indices


def compute_sensitivity(prediction_head, features, v, lambda_val=1.0, device=None):
    """Computes sensitivity of prediction head to direction v."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prediction_head.eval()
    features = features.to(device)
    v = v.to(device)
    
    with torch.no_grad():
        logits_orig = prediction_head(features)
        features_perturbed = features - lambda_val * v.unsqueeze(0)
        logits_perturbed = prediction_head(features_perturbed)
        
        diff = logits_orig - logits_perturbed
        l2_per_sample = torch.norm(diff, dim=1)
        avg_sensitivity = l2_per_sample.mean().item()
    
    return avg_sensitivity


def print_component_r2_table(combined_pls, features, labels, prediction_head, n_components=5):
    """
    Prints individual and cumulative R² for each component.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_dev = features.to(device)
    labels_dev = labels.to(device)
    prediction_head.eval()
    
    with torch.no_grad():
        logits = prediction_head(features_dev)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        losses = criterion(logits, labels_dev).cpu().numpy()
        
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        confidence = -entropy.cpu().numpy()
    
    features_np = features.cpu().numpy()
    components = combined_pls.get_components()
    scores = combined_pls.transform()
    
    n_show = min(n_components, components.shape[1])
    
    print("\n" + "-"*80)
    print(f"{'Comp':<6} {'Loss R²':<12} {'Conf R²':<12} {'Cumul Loss R²':<15} {'Cumul Conf R²':<15}")
    print("-"*80)
    
    cumul_loss_r2 = 0
    cumul_conf_r2 = 0
    
    for i in range(n_show):
        vec = components[:, i]
        vec = vec / np.linalg.norm(vec)
        proj = features_np @ vec
        
        loss_r2 = compute_variance_explained(proj, losses)
        conf_r2 = compute_variance_explained(proj, confidence)
        
        # Cumulative R² using all components up to i
        # This uses the PLS scores which are orthogonal
        cumul_proj_loss = scores[:, :i+1]
        cumul_proj_conf = scores[:, :i+1]
        
        if i == 0:
            cumul_loss_r2 = loss_r2
            cumul_conf_r2 = conf_r2
        else:
            # Fit multivariate regression for cumulative
            reg_loss = LinearRegression()
            reg_loss.fit(cumul_proj_loss, losses.reshape(-1, 1))
            pred_loss = reg_loss.predict(cumul_proj_loss)
            ss_res_loss = np.sum((losses.reshape(-1, 1) - pred_loss) ** 2)
            ss_tot_loss = np.sum((losses - losses.mean()) ** 2)
            cumul_loss_r2 = 1 - (ss_res_loss / ss_tot_loss) if ss_tot_loss > 0 else 0
            
            reg_conf = LinearRegression()
            reg_conf.fit(cumul_proj_conf, confidence.reshape(-1, 1))
            pred_conf = reg_conf.predict(cumul_proj_conf)
            ss_res_conf = np.sum((confidence.reshape(-1, 1) - pred_conf) ** 2)
            ss_tot_conf = np.sum((confidence - confidence.mean()) ** 2)
            cumul_conf_r2 = 1 - (ss_res_conf / ss_tot_conf) if ss_tot_conf > 0 else 0
        
        print(f"  {i+1:<4} {loss_r2:<12.4f} {conf_r2:<12.4f} {cumul_loss_r2:<15.4f} {cumul_conf_r2:<15.4f}")
    
    print("-"*80)




def evaluate_on_subset(prediction_head, features, labels, device):
    """Evaluates prediction head on a subset of features/labels."""
    prediction_head.eval()
    features_dev = features.to(device)
    labels_dev = labels.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = prediction_head(features_dev)
        loss = criterion(logits, labels_dev).item()
        _, predicted = torch.max(logits, 1)
        acc = (predicted == labels_dev).sum().item() / len(labels_dev)
    
    return loss, acc


def evaluate_percentile_subsets(prediction_head_baseline, prediction_head_clean, 
                                 val_features, val_features_clean, val_labels,
                                 direction, class_idx, comp_idx, device):
    """
    Evaluates on top 1%, 5%, 10% for a specific direction.
    """
    print(f"\n  --- Class {class_idx}, Component {comp_idx+1} ---")
    print(f"  {'Pct':<8} {'Base Loss':<11} {'Clean Loss':<11} {'Loss Δ%':<10} {'Base Acc':<10} {'Clean Acc':<10} {'Acc Δ':<8}")
    print("  " + "-"*70)
    
    for pct in [1, 5, 10]:
        indices = get_top_spurious_indices(val_features, direction, percentile=pct)
        n_samples = len(indices)
        
        if n_samples == 0:
            print(f"  Top {pct}%       No samples")
            continue
        
        subset_features = val_features[indices]
        subset_features_clean = val_features_clean[indices]
        subset_labels = val_labels[indices]
        
        base_loss, base_acc = evaluate_on_subset(prediction_head_baseline, subset_features, subset_labels, device)
        clean_loss, clean_acc = evaluate_on_subset(prediction_head_clean, subset_features_clean, subset_labels, device)
        
        loss_chg = ((clean_loss - base_loss) / base_loss) * 100 if base_loss > 0 else 0
        print(f"  Top {pct}%{'':<3} {base_loss:<11.4f} {clean_loss:<11.4f} {loss_chg:+8.1f}%  {base_acc*100:<10.2f} {clean_acc*100:<10.2f} {(clean_acc-base_acc)*100:+.2f}%")


def main(args):
    # Hyperparameters
    num_samples = 10000
    n_components = 5
    batch_size = 64
    
    print("="*60)
    print("  LINEAR CONCEPT ERASURE")
    print(f"  Dataset: {args.dataset}")
    print(f"  Mode: {args.mode}")
    print("="*60)
    
    # --- Step 1: Load Model ---
    print("\n--- 1. Loading Phikon Model and Processor ---")
    try:
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        phikon_model = AutoModel.from_pretrained("owkin/phikon")
        print("Phikon loaded successfully.")
    except Exception as e:
        print(f"Error loading Phikon: {e}")
        return

    # --- Step 2: Load Dataset ---
    print("\n--- 2. Loading Dataset ---")
    try:
        if args.dataset == 'chexpert':
            print("Loading CheXpert from HuggingFace datasets...")
            full_dataset = load_dataset("danjacobellis/chexpert", split='train')
            print(f"Loaded {len(full_dataset)} samples from CheXpert.")
            
            print("Converting multilabel to multiclass...")
            full_dataset = convert_chexpert_to_multiclass(full_dataset)
            
            chexpert_num_samples = min(50000, len(full_dataset))
            chexpert_val_samples = min(10000, len(full_dataset) - chexpert_num_samples)
            
            indices = list(range(chexpert_num_samples))
            train_subset = full_dataset.select(indices)
            
            # Oversampling for validation set
            print(f"Constructing CheXpert validation set with sickness rate: {args.chexpert_sickness_rate:.0%}")
            
            pool_start = chexpert_num_samples
            pool_indices = np.arange(pool_start, len(full_dataset))
            all_labels = np.array(full_dataset['label'])
            pool_labels = all_labels[pool_indices]
            
            pool_healthy = pool_indices[pool_labels == 0]
            pool_sick = pool_indices[pool_labels > 0]
            
            n_sick = min(int(chexpert_val_samples * args.chexpert_sickness_rate), len(pool_sick))
            n_healthy = min(chexpert_val_samples - n_sick, len(pool_healthy))
            
            selected_sick = np.random.choice(pool_sick, n_sick, replace=False)
            selected_healthy = np.random.choice(pool_healthy, n_healthy, replace=False)
            
            val_indices = np.concatenate([selected_sick, selected_healthy])
            np.random.shuffle(val_indices)
            
            print(f"  -> {len(val_indices)} samples: {n_sick} sick ({n_sick/len(val_indices):.1%}), {n_healthy} healthy")
            
            val_subset = full_dataset.select(val_indices.tolist())
            
            dataset = train_subset
            val_dataset = val_subset
            print(f"Using CheXpert dataset (14 classes).")
        else:
            # Load PCAM from HuggingFace
            print("Loading PCAM from HuggingFace datasets...")
            full_dataset = load_dataset("1aurent/PatchCamelyon", split='train')
            
            indices = list(range(num_samples))
            train_subset = full_dataset.select(indices)
            
            val_samples = 2000
            val_indices = list(range(num_samples, num_samples + val_samples))
            val_subset = full_dataset.select(val_indices)
            
            if args.dataset == 'normal':
                dataset = train_subset
                val_dataset = val_subset
                print("Using normal (unmodified) PCAM dataset.")
            elif args.dataset == 'spurious':
                artifact_probs = {0: 0.0, 1: 0.9}
                dataset = SpuriousPCAM(train_subset, artifact_probs=artifact_probs, square_size=30)
                val_dataset = SpuriousPCAM(val_subset, artifact_probs=artifact_probs, square_size=30)
                print("Using SpuriousPCAM (black square artifact).")
            elif args.dataset == 'rotation':
                artifact_probs = {0: 0.0, 1: 0.9}
                dataset = RotationSpuriousPCAM(train_subset, artifact_probs=artifact_probs)
                val_dataset = RotationSpuriousPCAM(val_subset, artifact_probs=artifact_probs)
                print("Using RotationSpuriousPCAM (90-degree rotation artifact).")
        
        print(f"Training samples: {len(dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- Step 3: Extract Features ---
    print("\n--- 3. Extracting Features ---")
    print("Extracting training features...")
    features, labels = extract_features(phikon_model, processor, dataset)
    print(f"Train Features shape: {features.shape}")
    
    print("Extracting validation features...")
    val_features, val_labels = extract_features(phikon_model, processor, val_dataset)
    print(f"Val Features shape: {val_features.shape}")

    # --- Step 4: Train Prediction Head ---
    print("\n--- 4. Training Prediction Head ---")
    prediction_head = train_prediction_head(
        features=features,
        labels=labels,
        hidden_dims=[256, 128],
        epochs=10,
        batch_size=64,
        val_features=val_features,
        val_labels=val_labels,
        dropout_rate=0.2,
        weight_decay=1e-5
    )
    print("Prediction head trained.")
    
    # --- Baseline Evaluation ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_loss, base_acc = evaluate_on_subset(prediction_head, val_features, val_labels, device)
    print(f"\nBaseline Val Loss: {base_loss:.4f}")
    print(f"Baseline Val Acc:  {base_acc*100:.2f}%")

    # --- Step 5: Per-Class PLS Analysis ---
    print("\n--- 5. Computing Per-Class Combined PLS ---")
    
    unique_classes = torch.unique(val_labels).tolist()
    pls_data = {}
    
    for c in unique_classes:
        print(f"\n{'='*50}")
        print(f"  CLASS {c}")
        print(f"{'='*50}")
        
        class_mask = (val_labels == c)
        features_c = val_features[class_mask]
        labels_c = val_labels[class_mask]
        
        print(f"Samples in Class {c}: {len(features_c)}")
        
        if len(features_c) < 10:
            print(f"Skipping class {c} (too few samples)")
            continue
        
        # Compute Combined PLS
        combined_pls_c = CombinedPLS(features_c, labels_c, prediction_head)
        combined_pls_c.fit(n_components=n_components)
        
        # Print R² table
        print_component_r2_table(combined_pls_c, features_c, labels_c, prediction_head, n_components)
        
        # Store
        pls_data[c] = {
            'combined_pls': combined_pls_c,
            'features': features_c,
            'labels': labels_c
        }

    # --- Step 6: Select Components to Remove ---
    directions_to_remove = []
    direction_info = []  # (class_idx, comp_idx, direction)
    
    if args.mode == 'interactive':
        print("\n" + "="*60)
        print("  INTERACTIVE MODE - SELECT COMPONENTS")
        print("="*60)
        
        # Select class
        available_classes = list(pls_data.keys())
        while True:
            try:
                selected_class = int(input(f"\nSelect class ({available_classes}): "))
                if selected_class in available_classes:
                    break
                print(f"Invalid. Choose from {available_classes}")
            except ValueError:
                print("Enter a valid integer.")
        
        # Show R² table again
        combined_pls = pls_data[selected_class]['combined_pls']
        features_c = pls_data[selected_class]['features']
        labels_c = pls_data[selected_class]['labels']
        print_component_r2_table(combined_pls, features_c, labels_c, prediction_head, n_components)
        
        # Select components
        while True:
            try:
                comp_input = input("\nEnter component indices to remove (comma-separated, e.g., '1,2,3'): ")
                comp_indices = [int(x.strip()) - 1 for x in comp_input.split(',')]
                if all(0 <= idx < n_components for idx in comp_indices):
                    break
                print(f"Invalid indices. Must be 1-{n_components}")
            except ValueError:
                print("Enter valid integers separated by commas.")
        
        # Collect directions
        for comp_idx in comp_indices:
            sign = combined_pls.align_sign()
            v = combined_pls.get_spurious_direction(component_idx=comp_idx) * sign
            directions_to_remove.append(v)
            direction_info.append((selected_class, comp_idx, v))
            print(f"  Added: Class {selected_class}, Component {comp_idx+1}")
    
    else:  # Automated mode
        print("\n" + "="*60)
        print("  AUTOMATED MODE - REMOVING TOP COMPONENT FROM EACH CLASS")
        print("="*60)
        
        for c in pls_data.keys():
            combined_pls = pls_data[c]['combined_pls']
            sign = combined_pls.align_sign()
            v = combined_pls.get_spurious_direction(component_idx=0) * sign
            directions_to_remove.append(v)
            direction_info.append((c, 0, v))
            print(f"  Added: Class {c}, Component 1")
    
    print(f"\nTotal directions to remove: {len(directions_to_remove)}")

    # --- Step 7: Project Out Directions ---
    print("\n--- 6. Applying Linear Concept Erasure ---")
    
    # Project out each direction sequentially (no orthogonalization needed)
    features_clean = features.clone()
    val_features_clean = val_features.clone()
    
    for i, v in enumerate(directions_to_remove):
        v = v.to(features.device)
        v = v / torch.norm(v)  # Normalize
        
        # Project out from training features
        proj_train = torch.matmul(features_clean, v)
        features_clean = features_clean - proj_train.unsqueeze(1) * v.unsqueeze(0)
        
        # Project out from validation features
        proj_val = torch.matmul(val_features_clean, v)
        val_features_clean = val_features_clean - proj_val.unsqueeze(1) * v.unsqueeze(0)
    
    print(f"Projected out {len(directions_to_remove)} directions from features.")

    # --- Step 8: Retrain Prediction Head ---
    print("\n--- 7. Retraining Prediction Head on Cleaned Features ---")
    prediction_head_clean = train_prediction_head(
        features=features_clean,
        labels=labels,
        hidden_dims=[256, 128],
        epochs=10,
        batch_size=batch_size,
        val_features=val_features_clean,
        val_labels=val_labels,
        dropout_rate=0.2,
        weight_decay=1e-5
    )

    # --- Step 9: Overall Evaluation ---
    print("\n--- 8. Overall Evaluation ---")
    clean_loss, clean_acc = evaluate_on_subset(prediction_head_clean, val_features_clean, val_labels, device)
    
    loss_pct_change = ((clean_loss - base_loss) / base_loss) * 100 if base_loss > 0 else 0
    
    print(f"Linear Erasure Val Loss: {clean_loss:.4f} (change: {clean_loss - base_loss:+.4f}, {loss_pct_change:+.2f}%)")
    print(f"Linear Erasure Val Acc:  {clean_acc*100:.2f}% (change: {(clean_acc - base_acc)*100:+.2f}%)")

    # --- Step 10: Per-Component Percentile Evaluation ---
    print("\n--- 9. Per-Component Spurious Subset Evaluation ---")
    
    for class_idx, comp_idx, direction in direction_info:
        evaluate_percentile_subsets(
            prediction_head, prediction_head_clean,
            val_features, val_features_clean, val_labels,
            direction, class_idx, comp_idx, device
        )

    # --- Step 11: Summary ---
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"{'Metric':<30} {'Baseline':<15} {'Linear Erasure':<15} {'Change':<15}")
    print("-"*75)
    print(f"{'Overall Val Loss':<30} {base_loss:<15.4f} {clean_loss:<15.4f} {clean_loss - base_loss:+.4f} ({loss_pct_change:+.2f}%)")
    print(f"{'Overall Val Acc (%)':<30} {base_acc*100:<15.2f} {clean_acc*100:<15.2f} {(clean_acc - base_acc)*100:+.2f}%")
    print("-"*75)
    print(f"Directions removed: {len(directions_to_remove)}")
    for class_idx, comp_idx, _ in direction_info:
        print(f"  - Class {class_idx}, Component {comp_idx+1}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Concept Erasure for Spurious Correlation Removal")
    parser.add_argument('--dataset', type=str, default='normal',
                        choices=['normal', 'spurious', 'rotation', 'chexpert'],
                        help='Dataset: normal, spurious (black square), rotation (90deg), chexpert')
    parser.add_argument('--chexpert-sickness-rate', type=float, default=0.3,
                        help='Target proportion of sick patients (Class > 0) in CheXpert validation set (default: 0.3)')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'automated'],
                        help='interactive: user selects class/components; automated: removes top component from each class')
    args = parser.parse_args()
    main(args)
