import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
import numpy as np

class LossPLS:
    """
    Performs Partial Least Squares (PLS) regression where:
    X = Feature set (e.g., Phikon embeddings)
    Y = CrossEntropyLoss of a prediction head evaluated on (X, labels)
    
    This essentially finds directions in the feature space that maximally explain 
    the variance in the model's loss (i.e., identifying features associated with hardness/easiness).
    """
    def __init__(self, features, labels, prediction_head, device=None):
        """
        Args:
            features: Tensor of shape (num_samples, feature_dim)
            labels: Tensor of shape (num_samples,)
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

    def _compute_loss_vector(self):
        """
        Computes the CrossEntropyLoss for each sample in the dataset.
        Returns:
            loss_vector: Numpy array of shape (num_samples, 1)
        """
        criterion = nn.CrossEntropyLoss(reduction='none') # important: no reduction to get per-sample loss
        
        with torch.no_grad():
            outputs = self.prediction_head(self.features)
            losses = criterion(outputs, self.labels)
            
        return losses.cpu().numpy().reshape(-1, 1)

    def _compute_target_vector(self):
        """
        Alias for _compute_loss_vector to support generic visualization interface.
        """
        return self._compute_loss_vector()

    def fit(self, n_components=2):
        """
        Fits the PLS regression model.
        
        Args:
            n_components: Number of components to keep.
        """
        # X: Features (convert to numpy)
        X = self.features.cpu().numpy()
        
        # Y: Per-sample Loss
        Y = self._compute_loss_vector()
        
        print(f"Fitting PLS with X shape {X.shape} and Y shape {Y.shape}...")
        
        self.pls_model = PLSRegression(n_components=n_components)
        self.pls_model.fit(X, Y)
        
        print("PLS fit complete.")
        
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

