import sys
import os
import torch
from transformers import AutoImageProcessor, AutoModel
from torchvision.datasets import PCAM
from torch.utils.data import Subset

# Add parent directory to path to allow imports from utils and PLS
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extraction import extract_features
from utils.prediction_head import train_prediction_head
from PLS.loss_pls import LossPLS
from PLS.confidence_pls import ConfidencePLS
from PLS.viz import PLSViz

def main():
    print("--- 1. Loading Phikon Model and Processor ---")
    try:
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        model = AutoModel.from_pretrained("owkin/phikon")
        print("Phikon loaded successfully.")
    except Exception as e:
        print(f"Error loading Phikon: {e}")
        return

    print("\n--- 2. Loading PatchCamelyon Dataset (via PyTorch) ---")
    try:
        # PCAM will be downloaded to ./data if not present
        data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_root, exist_ok=True)
        
        print(f"Downloading/Loading PCAM to {data_root}...")
        full_dataset = PCAM(root=data_root, split='train', download=True)
        
        # Subset for quick analysis
        num_samples = 10000
        indices = torch.arange(num_samples)
        dataset = Subset(full_dataset, indices)
        
        print(f"Loaded full dataset ({len(full_dataset)} samples).")
        print(f"Using subset of {len(dataset)} samples for analysis.")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\n--- 3. Extracting Features ---")
    features, labels = extract_features(model, processor, dataset)
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    print("\n--- 4. Training Prediction Head ---")
    # Using a simple MLP config
    prediction_head = train_prediction_head(
        features=features,
        labels=labels,
        hidden_dims=[256, 128],
        epochs=10, 
        batch_size=32
    )
    print("Prediction head trained.")

    # --- Per-Class PLS Analysis ---
    classes = torch.unique(labels)
    for c in classes:
        c_item = c.item()
        print(f"\n==========================================")
        print(f"   Analyzing Class {c_item}")
        print(f"==========================================")
        
        # Filter data for this class
        class_indices = (labels == c).nonzero(as_tuple=True)[0]
        print(f"Found {len(class_indices)} samples for Class {c_item}.")
        
        if len(class_indices) == 0:
            print(f"Skipping Class {c_item} (no samples).")
            continue
            
        features_c = features[class_indices]
        labels_c = labels[class_indices]
        
        # Create a Subset of the dataset for this class so indices align
        dataset_c = Subset(dataset, class_indices)
        
        print("\n--- Computing PLS Components on Loss ---")
        loss_pls = LossPLS(features_c, labels_c, prediction_head)
        loss_pls.fit(n_components=min(5, len(class_indices)//2)) # Adjust components if few samples
        
        print("\n--- Computing PLS Components on Confidence ---")
        conf_pls = ConfidencePLS(features_c, labels_c, prediction_head)
        conf_pls.fit(n_components=min(5, len(class_indices)//2))

        print("\n--- Generating Visualizations ---")
        
        # --- Loss Visualizations ---
        viz_loss = PLSViz(loss_pls, dataset_c)
        
        viz_loss.plot_predicted_vs_actual(target_name=f"Loss_Class{c_item}")
        viz_loss.plot_top_images(n_top=10, filename_suffix=f"_loss_class_{c_item}")

        # --- Confidence Visualizations ---
        viz_conf = PLSViz(conf_pls, dataset_c)
        
        viz_conf.plot_predicted_vs_actual(target_name=f"Confidence_negEntropy_Class{c_item}")
        viz_conf.plot_top_images(n_top=10, filename_suffix=f"_confidence_class_{c_item}")
        
        # --- Combined Visualizations ---
        PLSViz.plot_combined_top_images(dataset_c, loss_pls, conf_pls, n_components=3, n_top=10)
        # Rename the combined file to include class
        # (The method creates 'top_images_combined_components.png', we should rename/move it)
        default_combined_path = os.path.join("figures", "top_images_combined_components.png")
        new_combined_path = os.path.join("figures", f"top_images_combined_components_class_{c_item}.png")
        if os.path.exists(default_combined_path):
            os.rename(default_combined_path, new_combined_path)
            print(f"Saved combined top images plot to {new_combined_path}")

        # --- Similarity Visualizations ---
        similarity_filename = f"component_similarity_class_{c_item}.png"
        PLSViz.plot_component_similarity(loss_pls, conf_pls, filename=similarity_filename)

    print("\nVisualizations complete. Check the 'figures' directory.")

if __name__ == "__main__":
    main()
