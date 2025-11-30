import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import math

class PLSViz:
    def __init__(self, pls_instance, dataset):
        """
        Args:
            pls_instance: Instance of PLS.loss_pls.LossPLS or PLS.confidence_pls.ConfidencePLS. 
                          Must have been fitted.
                          Accesses .features, .prediction_head, .pls_model internally.
            dataset: The dataset object used to generate features. 
                     Must support indexing (dataset[i]) and return (image, label) 
                     or dictionary with 'image'.
        """
        self.pls = pls_instance
        self.dataset = dataset
        self.output_dir = "figures"
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_predicted_vs_actual(self, target_name="Loss"):
        """
        Plots predicted target (from PLS) vs actual target (from prediction head).
        Calculates and displays R^2.
        Saves to figures/predicted_vs_actual_{target_name.lower()}.png.
        """
        if self.pls.pls_model is None:
            raise ValueError(f"{type(self.pls).__name__} instance must be fitted before visualization.")

        # Get inputs
        X = self.pls.features.cpu().numpy()
        
        # Actual target (Y)
        # We access the generic method _compute_target_vector
        if hasattr(self.pls, '_compute_target_vector'):
            Y_actual = self.pls._compute_target_vector().flatten()
        elif hasattr(self.pls, '_compute_loss_vector'):
             # Fallback for old LossPLS if not updated (though we updated it)
             Y_actual = self.pls._compute_loss_vector().flatten()
        else:
            raise AttributeError("PLS instance must implement _compute_target_vector")
        
        # Predicted target (Y_hat)
        # PLS predict returns shape (n_samples, n_targets)
        Y_pred = self.pls.pls_model.predict(X).flatten()
        
        # Calculate R^2
        r2 = r2_score(Y_actual, Y_pred)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_actual, Y_pred, alpha=0.5, s=10)
        
        # Plot perfect prediction line
        min_val = min(Y_actual.min(), Y_pred.min())
        max_val = max(Y_actual.max(), Y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        plt.title(f'Predicted vs Actual {target_name} (R² = {r2:.4f})')
        plt.xlabel(f'Actual {target_name}')
        plt.ylabel(f'Predicted {target_name} (from PLS projection)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"predicted_vs_actual_{target_name.lower()}.png"
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {target_name} prediction plot to {save_path}")

    def plot_top_images(self, n_top=10, filename_suffix=""):
        """
        Plots the top 10 images that have the highest projection onto the first two components.
        Saves to figures/top_images_components{filename_suffix}.png.
        """
        if self.pls.pls_model is None:
            raise ValueError(f"{type(self.pls).__name__} instance must be fitted before visualization.")

        # Get scores (projections of images onto PLS components)
        # Shape: (n_samples, n_components)
        scores = self.pls.transform()
        
        n_available = scores.shape[1]
        n_components = min(n_available, 10)
        print(f"DEBUG: PLS scores shape: {scores.shape}. Plotting top {n_components} components.")
        
        if n_components == 0:
            print("No components found in PLS model.")
            return

        fig, axes = plt.subplots(n_components, n_top, figsize=(2 * n_top, 3 * n_components))
        # Handle case where n_components=1 (axes is 1D array)
        if n_components == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for comp_idx in range(n_components):
            if comp_idx >= scores.shape[1]:
                print(f"Warning: Skipping component {comp_idx} as it exceeds scores dimension {scores.shape[1]}")
                break

            # Get scores for this component
            comp_scores = scores[:, comp_idx]
            
            # Get indices of top scorers
            # argsort sorts ascending, so we take the last n_top and reverse
            top_indices = np.argsort(comp_scores)[-n_top:][::-1]
            
            for rank, idx in enumerate(top_indices):
                ax = axes[comp_idx, rank]
                
                # Fetch image from dataset
                sample = self.dataset[idx]
                # Handle different dataset formats
                if isinstance(sample, dict) and 'image' in sample:
                    img = sample['image']
                elif isinstance(sample, (tuple, list)):
                    img = sample[0]
                else:
                    # Fallback, maybe sample itself is the image
                    img = sample
                
                ax.imshow(img)
                ax.axis('off')
                if rank == 0:
                    ax.set_title(f"Comp {comp_idx+1}\nScore: {comp_scores[idx]:.2f}", fontsize=10)
                else:
                    ax.set_title(f"{comp_scores[idx]:.2f}", fontsize=10)

        plt.tight_layout()
        filename = f"top_images_components{filename_suffix}.png"
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved top images plot to {save_path}")

    @staticmethod
    def plot_combined_top_images(dataset, loss_pls, conf_pls, n_components=3, n_top=10):
        """
        Plots top 10 images scoring highest on sum of scores from (loss component, confidence component) pairs.
        Iterates through all combinations of the top `n_components` from both models.
        """
        loss_scores = loss_pls.transform()
        conf_scores = conf_pls.transform()
        
        n_loss = min(loss_scores.shape[1], n_components)
        n_conf = min(conf_scores.shape[1], n_components)
        
        n_combinations = n_loss * n_conf
        
        if n_combinations == 0:
             print("Not enough components for combined plotting.")
             return

        fig, axes = plt.subplots(n_combinations, n_top, figsize=(2 * n_top, 3 * n_combinations))
        # Handle case where n_combinations=1 (axes is 1D array)
        if n_combinations == 1:
            axes = np.expand_dims(axes, axis=0)

        plot_row = 0
        for l_idx in range(n_loss):
            for c_idx in range(n_conf):
                # Combined score: sum of scores on respective components
                # Normalize? For now raw sum as requested.
                combined_score = loss_scores[:, l_idx] + conf_scores[:, c_idx]
                
                # Get indices of top scorers
                top_indices = np.argsort(combined_score)[-n_top:][::-1]
                
                for rank, idx in enumerate(top_indices):
                    ax = axes[plot_row, rank]
                    
                    # Fetch image
                    sample = dataset[idx]
                    if isinstance(sample, dict) and 'image' in sample:
                        img = sample['image']
                    elif isinstance(sample, (tuple, list)):
                        img = sample[0]
                    else:
                        img = sample
                    
                    ax.imshow(img)
                    ax.axis('off')
                    
                    if rank == 0:
                        ax.set_title(f"L{l_idx+1} + C{c_idx+1}\nScore: {combined_score[idx]:.2f}", fontsize=10)
                    else:
                        ax.set_title(f"{combined_score[idx]:.2f}", fontsize=10)
                
                plot_row += 1

        plt.tight_layout()
        save_path = os.path.join("figures", "top_images_combined_components.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved combined top images plot to {save_path}")
    
    @staticmethod
    def plot_component_similarity(loss_pls, conf_pls, filename="component_similarity.png"):
        """
        Visualizes the pairwise cosine similarities between the loss components and the confidence components.
        """
        # Get components (weights in feature space)
        # Shape: (feature_dim, n_components) - Transpose because sklearn stores as (n_features, n_components)
        # Actually sklearn PLS x_weights_ is (n_features, n_components)
        # Let's double check sklearn docs or our wrapper.
        # wrapper .get_components() returns self.pls_model.x_weights_ which is (n_features, n_components)
        
        loss_comps = loss_pls.get_components()
        conf_comps = conf_pls.get_components()
        
        n_loss = loss_comps.shape[1]
        n_conf = conf_comps.shape[1]
        
        similarity_matrix = np.zeros((n_loss, n_conf))
        
        # Normalize columns to unit length for cosine similarity
        loss_comps_norm = loss_comps / np.linalg.norm(loss_comps, axis=0, keepdims=True)
        conf_comps_norm = conf_comps / np.linalg.norm(conf_comps, axis=0, keepdims=True)
        
        # Compute cosine similarity: A_norm.T @ B_norm
        similarity_matrix = np.dot(loss_comps_norm.T, conf_comps_norm)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        
        plt.title('Cosine Similarity between Loss and Confidence Components')
        plt.xlabel('Confidence Components')
        plt.ylabel('Loss Components')
        
        # Set ticks
        plt.xticks(np.arange(n_conf), [f'C{i+1}' for i in range(n_conf)])
        plt.yticks(np.arange(n_loss), [f'L{i+1}' for i in range(n_loss)])
        
        # Add text annotations
        for i in range(n_loss):
            for j in range(n_conf):
                text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join("figures", filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved component similarity heatmap to {save_path}")
