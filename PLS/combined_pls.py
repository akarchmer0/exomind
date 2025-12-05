import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import numpy as np


class CombinedPLS:
    """
    Performs Partial Least Squares (PLS) regression where:
    X = Feature set (e.g., Phikon embeddings)
    Y = [z-scored loss, z-scored confidence] (shape: N x 2)
    
    This directly finds directions in feature space that maximally explain 
    both loss AND confidence simultaneously - capturing "confidently wrong" patterns.
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
        self._loss_mean = None
        self._loss_std = None
        self._conf_mean = None
        self._conf_std = None

    def _compute_loss_vector(self):
        """
        Computes the CrossEntropyLoss for each sample in the dataset.
        Returns:
            loss_vector: Numpy array of shape (num_samples, 1)
        """
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            outputs = self.prediction_head(self.features)
            losses = criterion(outputs, self.labels)
            
        return losses.cpu().numpy().reshape(-1, 1)

    def _compute_confidence_vector(self):
        """
        Computes the NEGATIVE Entropy of the output distribution for each sample.
        Higher values = Lower Entropy = Higher Confidence.
        Returns:
            confidence_vector: Numpy array of shape (num_samples, 1)
        """
        with torch.no_grad():
            logits = self.prediction_head(self.features)
            probs = torch.softmax(logits, dim=1)
            eps = 1e-9
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
            negative_entropy = -entropy
            
        return negative_entropy.cpu().numpy().reshape(-1, 1)

    def _compute_combined_target(self):
        """
        Computes the combined target Y = [z-scored loss, z-scored confidence].
        Z-scoring ensures both targets contribute equally to PLS.
        Returns:
            Y: Numpy array of shape (num_samples, 2)
        """
        loss = self._compute_loss_vector()
        confidence = self._compute_confidence_vector()
        
        # Z-score normalization
        self._loss_mean = loss.mean()
        self._loss_std = loss.std()
        self._conf_mean = confidence.mean()
        self._conf_std = confidence.std()
        
        # Handle edge case of zero std
        loss_std = self._loss_std if self._loss_std > 1e-9 else 1.0
        conf_std = self._conf_std if self._conf_std > 1e-9 else 1.0
        
        loss_z = (loss - self._loss_mean) / loss_std
        conf_z = (confidence - self._conf_mean) / conf_std
        
        Y = np.hstack([loss_z, conf_z])
        return Y

    def _compute_target_vector(self):
        """
        Alias for _compute_combined_target to support generic visualization interface.
        Returns the combined z-scored target.
        """
        return self._compute_combined_target()

    def fit(self, n_components=5):
        """
        Fits the PLS regression model with multivariate target.
        
        Args:
            n_components: Number of components to keep.
        """
        X = self.features.cpu().numpy()
        Y = self._compute_combined_target()
        
        print(f"Fitting Combined PLS with X shape {X.shape} and Y shape {Y.shape}...")
        
        self.pls_model = PLSRegression(n_components=n_components)
        self.pls_model.fit(X, Y)
        
        print("Combined PLS fit complete.")
        
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

    def get_spurious_direction(self, component_idx=0):
        """
        Returns the specified component as the spurious direction.
        The first component (idx=0) captures the direction that explains
        both high loss AND high confidence (confidently wrong).
        
        Args:
            component_idx: Which component to use (default 0 = first component)
        
        Returns:
            v: Torch tensor of shape (feature_dim,), normalized to unit length
        """
        if self.pls_model is None:
            raise ValueError("PLS model has not been fitted yet.")
        
        components = self.get_components()
        v = components[:, component_idx]
        
        # Normalize to unit length
        v = v / np.linalg.norm(v)
        
        return torch.tensor(v, dtype=torch.float32)

    def get_variance_explained(self):
        """
        Computes variance explained by each component for both loss and confidence.
        
        Returns:
            dict with keys:
                'loss': array of R^2 for loss per component
                'confidence': array of R^2 for confidence per component
                'combined': array of average R^2 per component
        """
        if self.pls_model is None:
            raise ValueError("PLS model has not been fitted yet.")
        
        X = self.features.cpu().numpy()
        scores = self.pls_model.transform(X)
        
        loss = self._compute_loss_vector().flatten()
        confidence = self._compute_confidence_vector().flatten()
        
        n_components = scores.shape[1]
        loss_r2 = np.zeros(n_components)
        conf_r2 = np.zeros(n_components)
        
        for i in range(n_components):
            proj = scores[:, i]
            
            # R^2 for loss
            corr_loss = np.corrcoef(proj, loss)[0, 1]
            loss_r2[i] = corr_loss ** 2 if not np.isnan(corr_loss) else 0.0
            
            # R^2 for confidence
            corr_conf = np.corrcoef(proj, confidence)[0, 1]
            conf_r2[i] = corr_conf ** 2 if not np.isnan(corr_conf) else 0.0
        
        return {
            'loss': loss_r2,
            'confidence': conf_r2,
            'combined': (loss_r2 + conf_r2) / 2
        }

    def align_sign(self):
        """
        Determines if the first component needs sign flip.
        Positive projection should correspond to high loss AND high confidence.
        
        Returns:
            sign: +1 or -1
        """
        if self.pls_model is None:
            raise ValueError("PLS model has not been fitted yet.")
        
        X = self.features.cpu().numpy()
        scores = self.pls_model.transform(X)
        
        loss = self._compute_loss_vector().flatten()
        confidence = self._compute_confidence_vector().flatten()
        
        proj = scores[:, 0]
        
        corr_loss = np.corrcoef(proj, loss)[0, 1]
        corr_conf = np.corrcoef(proj, confidence)[0, 1]
        
        # We want positive projection = high loss AND high confidence
        # If both correlations are negative, flip sign
        # If mixed, use the product (both positive = keep, both negative = flip)
        if corr_loss * corr_conf > 0:
            # Same sign - check if positive
            if corr_loss > 0:
                return 1  # Already correct
            else:
                return -1  # Both negative, flip
        else:
            # Mixed signs - use loss as primary (high loss = spurious)
            if corr_loss > 0:
                return 1
            else:
                return -1

    def print_component_stats(self, n_components=5):
        """
        Prints statistics for each component showing variance explained
        for loss and confidence separately.
        """
        if self.pls_model is None:
            raise ValueError("PLS model has not been fitted yet.")
        
        var_explained = self.get_variance_explained()
        
        X = self.features.cpu().numpy()
        scores = self.pls_model.transform(X)
        
        loss = self._compute_loss_vector().flatten()
        confidence = self._compute_confidence_vector().flatten()
        
        print("\nCombined PLS Component Statistics:")
        print("-" * 70)
        print(f"{'Comp':<6} {'Loss Corr':<12} {'Loss R²':<12} {'Conf Corr':<12} {'Conf R²':<12}")
        print("-" * 70)
        
        n_to_show = min(n_components, scores.shape[1])
        for i in range(n_to_show):
            proj = scores[:, i]
            
            corr_loss = np.corrcoef(proj, loss)[0, 1]
            corr_conf = np.corrcoef(proj, confidence)[0, 1]
            
            print(f"  {i+1:<4} {corr_loss:+.4f}      {var_explained['loss'][i]:.4f}       "
                  f"{corr_conf:+.4f}      {var_explained['confidence'][i]:.4f}")
        
        print("-" * 70)

