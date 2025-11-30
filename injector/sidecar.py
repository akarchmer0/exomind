import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class SidecarModel(nn.Module):
    """
    A ResNet18-based sidecar model that outputs 768-dimensional vectors
    to match Phikon's embedding dimension.
    """
    def __init__(self, output_dim=768, pretrained=False):
        super(SidecarModel, self).__init__()
        # Load ResNet18 without pretrained weights
        self.backbone = resnet18(pretrained=pretrained)
        
        # Replace the final FC layer
        # ResNet18's fc layer input is 512
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, output_dim)
        
    def forward(self, x):
        return self.backbone(x)


class InjectorTrainer:
    """
    Trainer for the Sidecar Injector model.
    
    Trains the sidecar to adjust Phikon embeddings while penalizing
    reliance on a spurious direction v.
    """
    def __init__(self, prediction_head, sidecar_model, v, device=None):
        """
        Args:
            prediction_head: The frozen prediction head (MLP).
            sidecar_model: The SidecarModel to train.
            v: The spurious direction tensor (shape: (768,) or (1, 768)).
            device: 'cuda' or 'cpu'.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.prediction_head = prediction_head.to(self.device)
        self.prediction_head.eval()
        for param in self.prediction_head.parameters():
            param.requires_grad = False
            
        self.sidecar = sidecar_model.to(self.device)
        
        # Ensure v is a 1D tensor of shape (768,)
        self.v = v.to(self.device)
        if self.v.dim() == 2:
            self.v = self.v.squeeze(0)
        self.v = self.v / torch.norm(self.v)  # Normalize
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def train_epoch(self, dataloader, optimizer, lambda_val=1.0, alpha_penalty=0.1):
        """
        Trains the sidecar for one epoch.
        
        Args:
            dataloader: DataLoader yielding (images, h_phikon, labels).
            optimizer: Optimizer for sidecar parameters.
            lambda_val: Scaling factor for the spurious direction v.
            alpha_penalty: Weight for the invariance penalty term.
            
        Returns:
            avg_loss, avg_ce_loss, avg_penalty: Average losses for the epoch.
        """
        self.sidecar.train()
        
        total_loss = 0
        total_ce = 0
        total_penalty = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc="Training Sidecar")
        for batch in pbar:
            images, h_phikon, labels = batch
            images = images.to(self.device)
            h_phikon = h_phikon.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass through sidecar
            h_sidecar = self.sidecar(images)
            
            # Combined embedding
            h_combined = h_phikon + h_sidecar
            
            # Logits from prediction head
            logits_orig = self.prediction_head(h_combined)
            
            # Cross-entropy loss
            loss_ce = self.ce_loss(logits_orig, labels)
            
            # Invariance penalty
            # Perturb embedding by removing lambda * v
            h_perturbed = h_combined - lambda_val * self.v.unsqueeze(0)
            logits_perturbed = self.prediction_head(h_perturbed)
            
            # Penalize difference in logits
            penalty = self.mse_loss(logits_orig, logits_perturbed)
            
            # Total loss
            loss = loss_ce + alpha_penalty * penalty
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_penalty += penalty.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{loss_ce.item():.4f}",
                'penalty': f"{penalty.item():.4f}"
            })
        
        return total_loss / n_batches, total_ce / n_batches, total_penalty / n_batches
    
    def evaluate(self, dataloader):
        """
        Evaluates accuracy of the combined model (Phikon + Sidecar + Head).
        
        Args:
            dataloader: DataLoader yielding (images, h_phikon, labels).
            
        Returns:
            accuracy: Classification accuracy.
        """
        self.sidecar.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images, h_phikon, labels = batch
                images = images.to(self.device)
                h_phikon = h_phikon.to(self.device)
                labels = labels.to(self.device)
                
                h_sidecar = self.sidecar(images)
                h_combined = h_phikon + h_sidecar
                logits = self.prediction_head(h_combined)
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0

