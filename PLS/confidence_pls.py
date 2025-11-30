import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
import numpy as np

class ConfidencePLS:
    """
    Performs Partial Least Squares (PLS) regression where:
    X = Feature set (e.g., Phikon embeddings)
    Y = Entropy of the prediction head's output distribution (Confidence/Uncertainty)
    
    This finds directions in feature space that explain variance in model uncertainty.
    """
    def __init__(self, features, labels, prediction_head, device=None):
        """
        Args:
            features: Tensor of shape (num_samples, feature_dim)
            labels: Tensor of shape (num_samples,) - Not used for confidence calculation directly, but kept for API consistency
            prediction_head: A trained PyTorch model (e.g., MLPHead)
            device: 'cuda' or 'cpu'
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.prediction_head = prediction_head.to(self.device)
        self.prediction_head.eval()
        
        self.pls_model = None

    def _compute_target_vector(self):
        """
        Computes the NEGATIVE Entropy of the output distribution for each sample.
        Entropy = - sum(p * log(p))
        Target = -Entropy = sum(p * log(p))
        
        Higher values = Lower Entropy = Higher Confidence.
        """
        with torch.no_grad():
            logits = self.prediction_head(self.features)
            probs = torch.softmax(logits, dim=1)
            # Compute entropy: - sum(p * log(p))
            # Add epsilon to avoid log(0)
            eps = 1e-9
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
            
            # Return NEGATIVE entropy so that:
            # High Value = Low Entropy = High Confidence
            negative_entropy = -entropy
            
        return negative_entropy.cpu().numpy().reshape(-1, 1)

    def fit(self, n_components=2):
        """
        Fits the PLS regression model.
        
        Args:
            n_components: Number of components to keep.
        """
        # X: Features (convert to numpy)
        X = self.features.cpu().numpy()
        
        # Y: Per-sample Entropy
        Y = self._compute_target_vector()
        
        print(f"Fitting Confidence PLS with X shape {X.shape} and Y shape {Y.shape}...")
        
        self.pls_model = PLSRegression(n_components=n_components)
        self.pls_model.fit(X, Y)
        
        print("Confidence PLS fit complete.")
        
    def transform(self, features=None):
        """
        Projects features into the PLS latent space.
        
        Args:
            features: Optional tensor. If None, uses the training features.
        
        Returns:
            x_scores: The projected features (scores).
        """
        if self.pls_model is None:
            raise ValueError("PLS model has not been fitted yet. Call fit() first.")
            
        if features is None:
            X = self.features.cpu().numpy()
        else:
            X = features.cpu().numpy()
            
        return self.pls_model.transform(X)

    def get_components(self):
        """Returns the PLS weight vectors (directions in feature space)."""
        if self.pls_model is None:
            raise ValueError("PLS model has not been fitted yet.")
        return self.pls_model.x_weights_

