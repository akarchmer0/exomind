import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes=2, dropout_rate=0.1):
        super(MLPHead, self).__init__()
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_prediction_head(features, labels, hidden_dims=[256, 128], n_layers=None, 
                          batch_size=32, lr=1e-3, epochs=10, device=None,
                          val_features=None, val_labels=None):
    """
    Trains an MLP prediction head on the provided features and labels.

    Args:
        features: Tensor of shape (num_samples, feature_dim)
        labels: Tensor of shape (num_samples,)
        hidden_dims: List of integers defining the hidden layer dimensions.
        n_layers: Optional integer. If provided, checks if matches hidden_dims length.
        batch_size: Training batch size.
        lr: Learning rate.
        epochs: Number of training epochs.
        device: 'cuda' or 'cpu'.
        val_features: Optional validation features for early stopping.
        val_labels: Optional validation labels for early stopping.

    Returns:
        model: The trained MLPHead model (best validation loss if val provided).
    """
    if n_layers is not None and len(hidden_dims) != n_layers:
        print(f"Warning: n_layers ({n_layers}) does not match length of hidden_dims ({len(hidden_dims)}). Using hidden_dims.")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = features.shape[1]
    # Determine number of classes from labels
    num_classes = len(torch.unique(labels))
    
    print(f"Training MLP Head: Input Dim={input_dim}, Hidden Dims={hidden_dims}, Classes={num_classes}")

    model = MLPHead(input_dim, hidden_dims, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Ensure data is on the correct device or moved during iteration
    # Keeping dataset on CPU and moving batches to GPU is usually better for memory if dataset is large
    # but here features might fit in GPU memory. Let's keep them as tensor dataset.
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    
    # Prepare validation if provided
    has_validation = val_features is not None and val_labels is not None
    if has_validation:
        val_features_dev = val_features.to(device)
        val_labels_dev = val_labels.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_features, batch_labels in pbar:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (total / batch_size) if total > 0 else 0
            acc = 100 * correct / total if total > 0 else 0
            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{acc:.2f}%"})
        
        # Validation at end of epoch
        if has_validation:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_features_dev)
                val_loss = criterion(val_logits, val_labels_dev).item()
                _, val_predicted = torch.max(val_logits, 1)
                val_acc = (val_predicted == val_labels_dev).sum().item() / len(val_labels_dev)
            
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                print(f"  [New Best Model Saved]")
    
    # Load best model if validation was used
    if has_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val Loss: {best_val_loss:.4f}")
            
    return model

