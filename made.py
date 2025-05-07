import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# MADE Model with Order-Agnostic and Connectivity-Agnostic Training
class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(MADE, self).__init__()
        self.input_dim = input_dim  # Number of input dimensions (e.g., 784 for MNIST)
        self.hidden_dims = hidden_dims  # List of hidden layer sizes (e.g., [500, 500])
        self.dropout_rate = dropout_rate  # Dropout rate for regularization
        
        # Create lists for layers and related components
        self.layers = nn.ModuleList()  # Main network layers
        self.dropouts = nn.ModuleList()  # Dropout layers
        self.layer_norms = nn.ModuleList()  # Layer normalization layers
        self.residual_adapters = nn.ModuleList()  # Residual connection adapters
        prev_dim = input_dim
        
        # Build hidden layers
        for h_dim in hidden_dims:
            layer = nn.Linear(prev_dim, h_dim)
            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout_rate))
            self.layer_norms.append(nn.LayerNorm(h_dim))
            if prev_dim != h_dim:
                adapter = nn.Linear(prev_dim, h_dim)
                self.residual_adapters.append(adapter)
            else:
                self.residual_adapters.append(None)
            prev_dim = h_dim
        
        # Output layer (Bernoulli logits for binary data)
        self.output_layer = nn.Linear(prev_dim, input_dim)
        
        # Direct connections from input to output (as per the paper)
        self.direct_layer = nn.Linear(input_dim, input_dim, bias=False)
        
        # Initialize masks (will be updated during training)
        self.masks_W = []  # Masks for hidden layers
        self.mask_V = None  # Mask for output layer
        self.mask_A = None  # Mask for direct connections
        
    def create_masks(self, ordering=None):
        # Reset masks
        self.masks_W = []
        
        # If no ordering is provided, use natural ordering; otherwise, use the permuted ordering
        if ordering is None:
            ordering = torch.arange(self.input_dim)
        else:
            ordering = torch.tensor(ordering, dtype=torch.long)
        
        # Input connectivity: m^0(d) is the position of dimension d in the ordering
        m_prev = ordering  # Shape: (input_dim,)
        
        # Create masks for each hidden layer
        for l, h_dim in enumerate(self.hidden_dims):
            # Sample connectivity for hidden units: m^l(k) in [min(m^{l-1}), D-1]
            min_prev = m_prev.min().item()
            m_curr = torch.randint(min_prev, self.input_dim, (h_dim,))  # Shape: (h_dim,)
            
            # Mask for W^l: M_{k,d} = 1 if m^l(k) >= m^{l-1}(d)
            mask_W = (m_curr.unsqueeze(1) >= m_prev.unsqueeze(0)).float()  # Shape: (h_dim, prev_dim)
            self.masks_W.append(mask_W)
            
            m_prev = m_curr
        
        # Mask for output layer: M_{d,k} = 1 if ordering[d] > m^L(k)
        self.mask_V = (ordering.unsqueeze(1) > m_prev.unsqueeze(0)).float()  # Shape: (input_dim, h_dim)
        
        # Mask for direct connections: strictly lower diagonal
        self.mask_A = torch.tril(torch.ones(self.input_dim, self.input_dim), diagonal=-1)  # Shape: (input_dim, input_dim)
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape input: (batch_size, input_dim)
        h = x
        
        # Apply hidden layers
        for i in range(len(self.layers)):
            # Apply mask to weights
            masked_weight = self.layers[i].weight * self.masks_W[i].to(x.device)
            h = F.linear(h, masked_weight, self.layers[i].bias)  # Linear transformation
            h = F.relu(h)  # ReLU activation (as per paper's experiments)
            h = self.dropouts[i](h)  # Apply dropout
            h = self.layer_norms[i](h)  # Apply layer normalization
            
            # Residual connection
            if i % 2 == 0:
                residual = x if i == 0 else h_prev
                if self.residual_adapters[i] is not None:
                    residual = self.residual_adapters[i](residual)
                h = h + residual
            h_prev = h
        
        # Output layer
        masked_output_weight = self.output_layer.weight * self.mask_V.to(x.device)
        logits = F.linear(h, masked_output_weight, self.output_layer.bias)  # Shape: (batch_size, input_dim)
        
        # Direct connections
        masked_direct_weight = self.direct_layer.weight * self.mask_A.to(x.device)
        logits += F.linear(x, masked_direct_weight)  # Add direct connections
        
        return logits  # Return logits for Bernoulli distribution

# Training function with order-agnostic and connectivity-agnostic training
def train_made(model, data, lr=0.01, epochs=1, batch_size=100, device='cpu', num_masks=32, patience=30):
    model = model.to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)  # Adagrad as per paper
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    
    losses = []  # Track average NLL per epoch
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # Precompute a fixed number of masks for cycling (connectivity-agnostic training)
    mask_orderings = [np.random.permutation(model.input_dim) for _ in range(num_masks)]
    mask_idx = 0
    
    for epoch in range(epochs):
        model.train()
        # Order-agnostic: cycle through precomputed orderings
        ordering = mask_orderings[mask_idx]
        model.create_masks(ordering)  # Update masks with new ordering
        mask_idx = (mask_idx + 1) % num_masks  # Cycle through masks
        
        perm = torch.randperm(data_tensor.size(0))  # Shuffle data
        total_loss = 0
        for i in range(0, data_tensor.size(0), batch_size):
            indices = perm[i:i+batch_size]
            batch = data_tensor[indices]
            
            optimizer.zero_grad()
            logits = model(batch)  # Forward pass
            # Binary cross-entropy loss (negative log-likelihood)
            loss = F.binary_cross_entropy_with_logits(logits, batch, reduction='sum') / batch.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (data_tensor.size(0) // batch_size)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, NLL: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    return losses, mask_orderings

# Function to compute test NLL with multiple orderings
def evaluate_made(model, data, mask_orderings, device='cpu'):
    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    total_nll = 0
    
    with torch.no_grad():
        for ordering in mask_orderings:
            model.create_masks(ordering)
            logits = model(data_tensor)
            nll = F.binary_cross_entropy_with_logits(logits, data_tensor, reduction='sum')
            total_nll += nll.item()
    
    avg_nll = total_nll / len(mask_orderings) / len(data)
    return avg_nll

# Function to sample from the model
def sample_made(model, n_samples, ordering, device='cpu'):
    model.eval()
    samples = torch.zeros(n_samples, model.input_dim).to(device)
    
    with torch.no_grad():
        model.create_masks(ordering)
        for d in range(model.input_dim):
            logits = model(samples)
            probs = torch.sigmoid(logits[:, d])
            samples[:, d] = (torch.rand(n_samples, device=device) < probs).float()
    
    return samples.cpu().numpy()

# Visualize sampled digits (for MNIST)
def plot_samples(samples, filename, grid_size=(10, 10)):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    idx = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if idx < len(samples):
                axes[i, j].imshow(samples[idx].reshape(28, 28), cmap='gray')
                axes[i, j].axis('off')
                idx += 1
    plt.savefig(filename)
    plt.close()

# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Placeholder for dataset loading (e.g., UCI or MNIST)
    # For now, we'll simulate a binary dataset
    input_dim = 784  # Example: MNIST (28x28)
    n_train = 50000
    n_test = 10000
    train_data = np.random.randint(0, 2, (n_train, input_dim)).astype(np.float32)  # Simulated binary data
    test_data = np.random.randint(0, 2, (n_test, input_dim)).astype(np.float32)
    
    # Model configuration (as per paper's experiments)
    configs = [
        {"hidden_dims": [8000], "name": "1hl_8000units"},  # 1 hidden layer
        {"hidden_dims": [8000, 8000], "name": "2hl_8000units"},  # 2 hidden layers
    ]
    
    for config in configs:
        print(f"\nTraining with {config['name']}")
        model = MADE(input_dim=input_dim, hidden_dims=config["hidden_dims"])
        losses, mask_orderings = train_made(model, train_data, num_masks=32, device=device)
        
        # Evaluate test NLL
        test_nll = evaluate_made(model, test_data, mask_orderings, device=device)
        print(f"Test NLL ({config['name']}): {test_nll:.4f}")
        
        # Sample and visualize (for MNIST)
        samples = sample_made(model, 100, mask_orderings[0], device=device)
        plot_samples(samples, f"samples_{config['name']}.png")

if __name__ == "__main__":
    main()