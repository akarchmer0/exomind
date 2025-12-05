import sys
import os
import argparse
import torch
from transformers import AutoImageProcessor, AutoModel
from torchvision.datasets import PCAM
from torch.utils.data import Subset
from datasets import load_dataset

# Add parent directory to path to allow imports from utils and PLS
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extraction import extract_features
from utils.prediction_head import train_prediction_head
from PLS.loss_pls import LossPLS
from PLS.confidence_pls import ConfidencePLS
from PLS.combined_pls import CombinedPLS
from PLS.viz import PLSViz
from spurious_utils import SpuriousPCAM


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


def main():
    parser = argparse.ArgumentParser(description="Run PLS Diagnostic Analysis")
    parser.add_argument('--dataset', type=str, default='pcam',
                        choices=['pcam', 'kather', 'chexpert'],
                        help='Dataset: pcam (binary), kather (multiclass), chexpert (chest xray)')
    args = parser.parse_args()
    print("--- 1. Loading Phikon Model and Processor ---")
    try:
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        model = AutoModel.from_pretrained("owkin/phikon")
        print("Phikon loaded successfully.")
    except Exception as e:
        print(f"Error loading Phikon: {e}")
        return

    print(f"\n--- 2. Loading Dataset ({args.dataset.upper()}) ---")
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
                num_samples = min(100000, len(full_dataset))  # Use up to 20k samples
                analysis_samples = min(20000, len(full_dataset) - num_samples)
                
                train_indices = list(range(num_samples))
                analysis_indices = list(range(num_samples, num_samples + analysis_samples))
                
                train_subset = full_dataset.select(train_indices)
                analysis_subset = full_dataset.select(analysis_indices)
                
                train_dataset = train_subset
                analysis_dataset = analysis_subset
                
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
            num_samples = min(20000, len(full_dataset))  # Use up to 20k samples
            analysis_samples = min(4000, len(full_dataset) - num_samples)
            
            train_indices = list(range(num_samples))
            analysis_indices = list(range(num_samples, num_samples + analysis_samples))
            
            train_subset = full_dataset.select(train_indices)
            analysis_subset = full_dataset.select(analysis_indices)
            
            train_dataset = train_subset
            analysis_dataset = analysis_subset
            
            print(f"Using Kather100K dataset (9 tissue classes).")
            print(f"  Classes: 0-8 (Adipose, Background, Debris, Lymphocytes, Mucus, Smooth muscle, Normal colon, Cancer-associated stroma, Cancer epithelium)")
        else:
            # PCAM will be downloaded to ./data if not present
            data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(data_root, exist_ok=True)
            
            print(f"Downloading/Loading PCAM to {data_root}...")
            full_dataset = PCAM(root=data_root, split='train', download=True)
            
            # Split indices: Train (10k) vs Analysis (separate 2k)
            train_indices = torch.arange(0, 10000)
            analysis_indices = torch.arange(10000, 12000)
            
            train_subset = Subset(full_dataset, train_indices)
            analysis_subset = Subset(full_dataset, analysis_indices)
            
            # --- Create Spurious Datasets ---
            
            # TRAINING SET: Spurious Correlation
            # Artifact ONLY on Class 1 (50% of the time). Class 0 gets nothing.
            print("Creating Spurious Training Set (Artifact only on Class 1)...")
            train_dataset = SpuriousPCAM(
                train_subset, 
                artifact_probs={0: 0.0, 1: 0.5},  # <--- The Spurious Correlation
                square_size=40
            )

            # ANALYSIS SET: Balanced Artifact
            # Artifact on BOTH classes (50% of the time).
            print("Creating Balanced Analysis Set (Artifact on both classes)...")
            analysis_dataset = SpuriousPCAM(
                analysis_subset,
                artifact_probs={0: 0.5, 1: 0.5},  # <--- Broken Correlation
                square_size=40
            )

        print(f"Loaded full dataset ({len(full_dataset) if 'full_dataset' in locals() else 'N/A'} samples).")
        print(f"Using {len(train_dataset)} samples for training.")
        print(f"Using {len(analysis_dataset)} samples for analysis.")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\n--- 3. Extracting Features for Training ---")
    train_features, train_labels = extract_features(model, processor, train_dataset)
    print(f"Train features shape: {train_features.shape}")
    print(f"Train labels shape: {train_labels.shape}")

    print("\n--- 4. Training Prediction Head ---")
    # Using a simple MLP config
    prediction_head = train_prediction_head(
        features=train_features,
        labels=train_labels,
        hidden_dims=[512, 256, 128],
        epochs=20, 
        batch_size=64
    )
    print("Prediction head trained.")

    print("\n--- 5. Extracting Features for PLS Analysis ---")
    pls_features, pls_labels = extract_features(model, processor, analysis_dataset)
    print(f"Analysis features shape: {pls_features.shape}")
    print(f"Analysis labels shape: {pls_labels.shape}")

    # --- Per-Class PLS Analysis ---
    classes = torch.unique(pls_labels)
    for c in classes:
        c_item = c.item()
        print(f"\n==========================================")
        print(f"   Analyzing Class {c_item}")
        print(f"==========================================")
        
        # Filter data for this class
        class_indices = (pls_labels == c).nonzero(as_tuple=True)[0]
        print(f"Found {len(class_indices)} samples for Class {c_item}.")
        
        if len(class_indices) == 0:
            print(f"Skipping Class {c_item} (no samples).")
            continue
            
        features_c = pls_features[class_indices]
        labels_c = pls_labels[class_indices]
        
        # Create a Subset of the dataset for this class so indices align
        if hasattr(analysis_dataset, 'select'):
            # HuggingFace dataset
            dataset_c = analysis_dataset.select(class_indices.tolist())
        else:
            # PyTorch Subset
            dataset_c = Subset(analysis_dataset, class_indices)
        
        n_comps = min(5, len(class_indices)//2)
        
        print("\n--- Computing PLS Components on Loss ---")
        loss_pls = LossPLS(features_c, labels_c, prediction_head)
        loss_pls.fit(n_components=n_comps)
        
        print("\n--- Computing PLS Components on Confidence ---")
        conf_pls = ConfidencePLS(features_c, labels_c, prediction_head)
        conf_pls.fit(n_components=n_comps)
        
        print("\n--- Computing Combined PLS (Loss + Confidence) ---")
        combined_pls = CombinedPLS(features_c, labels_c, prediction_head)
        combined_pls.fit(n_components=n_comps)
        combined_pls.print_component_stats(n_components=n_comps)

        print("\n--- Generating Visualizations ---")
        
        # --- Loss Visualizations ---
        viz_loss = PLSViz(loss_pls, dataset_c)
        
        viz_loss.plot_predicted_vs_actual(target_name=f"Loss_Class{c_item}")
        viz_loss.plot_top_images(n_top=10, filename_suffix=f"_loss_class_{c_item}")

        # --- Confidence Visualizations ---
        viz_conf = PLSViz(conf_pls, dataset_c)
        
        viz_conf.plot_predicted_vs_actual(target_name=f"Confidence_negEntropy_Class{c_item}")
        viz_conf.plot_top_images(n_top=10, filename_suffix=f"_confidence_class_{c_item}")
        
        # --- Combined PLS Visualizations ---
        viz_combined = PLSViz(combined_pls, dataset_c)
        
        viz_combined.plot_predicted_vs_actual(target_name=f"Combined_Class{c_item}")
        viz_combined.plot_top_images(n_top=10, filename_suffix=f"_combined_class_{c_item}")

        # --- Similarity Visualizations ---
        similarity_filename = f"component_similarity_class_{c_item}.png"
        PLSViz.plot_component_similarity(loss_pls, conf_pls, filename=similarity_filename)

    print("\nVisualizations complete. Check the 'figures' directory.")

if __name__ == "__main__":
    main()
