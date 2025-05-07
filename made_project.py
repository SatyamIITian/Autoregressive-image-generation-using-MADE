import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to generate a spiral dataset
def make_spiral(n_samples, noise=0.05):
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    r = theta + np.random.randn(n_samples) * noise
    data = np.array([np.cos(theta) * r, np.sin(theta) * r]).T
    return data

# Generate datasets with custom scaling
def generate_datasets(n_samples=1000):
    datasets = {}
    scalers = {}
    
    # Moons
    moons = make_moons(n_samples=n_samples, noise=0.05, random_state=42)[0]
    scaler_moons = MinMaxScaler(feature_range=(-1, 1))
    moons_scaled = scaler_moons.fit_transform(moons)
    datasets["moons"] = moons_scaled
    scalers["moons"] = scaler_moons
    
    # Blobs
    blobs, blob_labels = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.5, random_state=42, return_centers=False)
    scaler_blobs = MinMaxScaler(feature_range=(-1, 1))
    blobs_scaled = scaler_blobs.fit_transform(blobs)
    datasets["blobs"] = blobs_scaled
    scalers["blobs"] = scaler_blobs
    datasets["blobs_labels"] = blob_labels
    
    # Spirals
    spirals = make_spiral(n_samples=n_samples, noise=0.05)
    scaler_spirals = MinMaxScaler(feature_range=(-1, 1))
    spirals_scaled = scaler_spirals.fit_transform(spirals)
    datasets["spirals"] = spirals_scaled
    scalers["spirals"] = scaler_spirals
    
    return datasets, scalers

# MADE Model with Residual Connections and Layer Normalization
class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, n_components=3, dropout_rate=0.2):
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.n_components = n_components
        
        # Create main layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_adapters = nn.ModuleList()  # Separate list for residual adapters
        prev_dim = input_dim
        prev_order = torch.arange(input_dim)
        
        for h_dim in hidden_dims:
            layer = nn.Linear(prev_dim, h_dim)
            # Create mask for hidden layer
            mask = torch.zeros(h_dim, prev_dim)
            curr_order = torch.randperm(h_dim) % input_dim
            for j in range(h_dim):
                mask[j, prev_order < curr_order[j]] = 1
            layer.weight.data *= mask
            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout_rate))
            self.layer_norms.append(nn.LayerNorm(h_dim))
            # Add residual adapter if dimensions don't match
            if prev_dim != h_dim:
                adapter = nn.Linear(prev_dim, h_dim)
                self.residual_adapters.append(adapter)
            else:
                self.residual_adapters.append(None)  # No adapter needed if dimensions match
            prev_dim = h_dim
            prev_order = curr_order
        
        # Output layer for GMM parameters
        output_layer = nn.Linear(prev_dim, output_dim)
        mask = torch.zeros(output_dim, prev_dim)
        for d in range(2):  # For each dimension (x, y)
            for c in range(n_components):
                mean_idx = d * n_components * 3 + c
                log_std_idx = d * n_components * 3 + n_components + c
                weight_idx = d * n_components * 3 + 2 * n_components + c
                mask[mean_idx, prev_order <= d] = 1
                mask[log_std_idx, prev_order <= d] = 1
                mask[weight_idx, prev_order <= d] = 1
        output_layer.weight.data *= mask
        self.layers.append(output_layer)
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for i in range(len(self.layers) - 1):
            residual = x
            x = self.layers[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
            x = self.layer_norms[i](x)
            # Apply residual connection
            if i % 2 == 0:
                if self.residual_adapters[i] is not None:
                    residual = self.residual_adapters[i](residual)
                x = x + residual
        x = self.layers[-1](x)
        return x

# Training function with advanced techniques
def train_made(model, data, lr=0.0001, epochs=300, batch_size=128, device='cpu', patience=50):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    
    losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(data_tensor.size(0))
        total_loss = 0
        for i in range(0, data_tensor.size(0), batch_size):
            indices = perm[i:i+batch_size]
            batch = data_tensor[indices]
            
            optimizer.zero_grad()
            output = model(batch)
            # Split output into GMM parameters (means, log_stds, weights) for each dimension
            n_components = model.n_components
            means = []
            log_stds = []
            weights = []
            for d in range(2):
                start = d * n_components * 3
                mean = output[:, start:start+n_components]
                log_std = output[:, start+n_components:start+2*n_components]
                weight = output[:, start+2*n_components:start+3*n_components]
                means.append(mean)
                log_stds.append(log_std)
                weights.append(weight)
            
            # Compute GMM negative log-likelihood
            nll_loss = 0
            for d in range(2):
                component_log_probs = -0.5 * ((batch[:, d].unsqueeze(1) - means[d]) / torch.exp(log_stds[d]))**2 - log_stds[d] - 0.5 * np.log(2 * np.pi)
                weights_d = F.softmax(weights[d], dim=1)
                log_prob = torch.logsumexp(component_log_probs + torch.log(weights_d + 1e-10), dim=1)
                nll_loss -= torch.mean(log_prob)
            
            # KL divergence regularization to encourage diverse components
            kl_loss = 0
            for d in range(2):
                weights_d = F.softmax(weights[d], dim=1)
                kl_loss += torch.mean(-torch.sum(weights_d * torch.log(weights_d + 1e-10), dim=1))
            
            loss = nll_loss + 0.01 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / (data_tensor.size(0) // batch_size)
        losses.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
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
    
    return losses

# Sampling function for GMM with annealed temperature
def sample_made(model, n_samples, device='cpu', temperature_start=0.8, temperature_end=1.2, min_log_std=-0.5):
    model.eval()
    samples = torch.zeros(n_samples, 2).to(device)
    n_components = model.n_components
    
    with torch.no_grad():
        for d in range(2):
            output = model(samples)
            start = d * n_components * 3
            means = output[:, start:start+n_components]
            log_stds = output[:, start+n_components:start+2*n_components]
            weights = output[:, start+2*n_components:start+3*n_components]
            
            # Apply minimum log_std
            log_stds = torch.clamp(log_stds, min=min_log_std)
            weights = F.softmax(weights, dim=1)
            
            # Sample component based on weights
            component_probs = weights / weights.sum(dim=1, keepdim=True)
            components = torch.multinomial(component_probs, 1).squeeze()
            
            # Annealed temperature
            temperature = temperature_start + (temperature_end - temperature_start) * (d / 1.0)
            
            # Sample from the selected component
            mean = means[torch.arange(n_samples), components]
            log_std = log_stds[torch.arange(n_samples), components]
            samples[:, d] = mean + torch.exp(log_std) * torch.randn(n_samples).to(device) * temperature
    
    return samples.cpu().numpy()

# Function to compute mean distance
def compute_mean_distance(original, generated):
    distances = np.sqrt(np.sum((original - generated) ** 2, axis=1))
    return np.mean(distances)

# Function to compute Wasserstein distance over epochs
def compute_wasserstein_distance(original, generated):
    w_dist_x = wasserstein_distance(original[:, 0], generated[:, 0])
    w_dist_y = wasserstein_distance(original[:, 1], generated[:, 1])
    return (w_dist_x + w_dist_y) / 2

# Enhanced visualization with W-Dist inset plot
def plot_results(original, generated, dataset_name, title, filename, scaler, datasets, losses, w_dists):
    original_orig = scaler.inverse_transform(original)
    generated_orig = scaler.inverse_transform(generated)
    
    x_min = min(original_orig[:, 0].min(), generated_orig[:, 0].min()) - 1
    x_max = max(original_orig[:, 0].max(), generated_orig[:, 0].max()) + 1
    y_min = min(original_orig[:, 1].min(), generated_orig[:, 1].min()) - 1
    y_max = max(original_orig[:, 1].max(), generated_orig[:, 1].max()) + 1
    
    mean_dist = compute_mean_distance(original_orig, generated_orig)
    w_dist = compute_wasserstein_distance(original_orig, generated_orig)
    final_loss = losses[-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#f0f0f0')
        ax.grid(True, alpha=0.3)
    
    if dataset_name == "moons":
        t = np.linspace(0, np.pi, 100)
        x1 = 1 + np.cos(t)
        y1 = np.sin(t)
        x2 = -np.cos(t)
        y2 = np.sin(t)
        for ax in [ax1, ax2]:
            ax.plot(x1, y1, 'k--', linewidth=2, label='Expected Shape')
            ax.plot(x2, y2, 'k--', linewidth=2)
            ax.fill(x1, y1, 'gray', alpha=0.1)
            ax.fill(x2, y2, 'gray', alpha=0.1)
        
        ax1.scatter(original_orig[:, 0], original_orig[:, 1], c='blue', s=40, alpha=0.8, label='Original Points')
        ax2.scatter(generated_orig[:, 0], generated_orig[:, 1], c='red', s=30, alpha=0.8, label='Generated Points')
    
    elif dataset_name == "blobs":
        labels = datasets.get("blobs_labels", None)
        colors = ['blue', 'green', 'purple']
        markers = ['o', '^', 's']
        
        if labels is not None:
            for cluster in range(3):
                mask = labels == cluster
                ax1.scatter(original_orig[mask, 0], original_orig[mask, 1], c=colors[cluster], marker=markers[cluster], s=60, alpha=0.8, label=f'Cluster {cluster}')
            kmeans = KMeans(n_clusters=3, random_state=42).fit(original_orig)
            centroids = kmeans.cluster_centers_
            ax1.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')
            for center in centroids:
                ellipse = Ellipse(xy=center, width=1.5, height=1.5, edgecolor='black', fc='None', lw=1, alpha=0.5)
                ax1.add_patch(ellipse)
        
        kmeans = KMeans(n_clusters=3, random_state=42).fit(generated_orig)
        gen_labels = kmeans.labels_
        for cluster in range(3):
            mask = gen_labels == cluster
            ax2.scatter(generated_orig[mask, 0], generated_orig[mask, 1], c=colors[cluster], marker=markers[cluster], s=50, alpha=0.8, label=f'Cluster {cluster}')
        centroids = kmeans.cluster_centers_
        ax2.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')
        for center in centroids:
            ellipse = Ellipse(xy=center, width=1.5, height=1.5, edgecolor='black', fc='None', lw=1, alpha=0.5)
            ax2.add_patch(ellipse)
    
    elif dataset_name == "spirals":
        theta = np.linspace(0, 2 * np.pi * 2, 1000)
        r = theta
        x_spiral = np.cos(theta) * r
        y_spiral = np.sin(theta) * r
        
        for ax in [ax1, ax2]:
            ax.plot(x_spiral, y_spiral, 'k--', linewidth=2, label='Expected Spiral')
        
        angles_orig = np.arctan2(original_orig[:, 1], original_orig[:, 0])
        colors_orig = (angles_orig + np.pi) / (2 * np.pi)
        angles_gen = np.arctan2(generated_orig[:, 1], generated_orig[:, 0])
        colors_gen = (angles_gen + np.pi) / (2 * np.pi)
        
        ax1.scatter(original_orig[:, 0], original_orig[:, 1], c=colors_orig, cmap='Blues', s=40, alpha=0.8, label='Original Points')
        ax2.scatter(generated_orig[:, 0], generated_orig[:, 1], c=colors_gen, cmap='Reds', s=30, alpha=0.8, label='Generated Points')
    
    # Set titles, labels, and limits
    ax1.set_title(f"Original {dataset_name.capitalize()} Data", fontsize=12)
    ax1.set_xlabel("X", fontsize=10)
    ax1.set_ylabel("Y", fontsize=10)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(fontsize=8)
    
    ax2.set_title(f"Generated {dataset_name.capitalize()} Data", fontsize=12)
    ax2.set_xlabel("X", fontsize=10)
    ax2.set_ylabel("Y", fontsize=10)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.legend(fontsize=8)
    
    # Add metrics annotation to generated subplot
    metrics_text = f"Mean Dist: {mean_dist:.2f}\nFinal Loss: {final_loss:.2f}\nW-Dist: {w_dist:.2f}"
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Inset loss plot
    inset_ax1 = ax2.inset_axes([0.65, 0.05, 0.3, 0.2])
    inset_ax1.plot(losses, color='purple', linewidth=1)
    inset_ax1.set_title("Loss Curve", fontsize=8)
    inset_ax1.set_xlabel("Epoch", fontsize=6)
    inset_ax1.set_ylabel("Loss", fontsize=6)
    inset_ax1.tick_params(axis='both', which='major', labelsize=5)
    inset_ax1.grid(True, alpha=0.3)
    
    # Inset W-Dist plot (last 50 epochs)
    inset_ax2 = ax2.inset_axes([0.65, 0.3, 0.3, 0.2])
    w_dist_subset = w_dists[-50:] if len(w_dists) >= 50 else w_dists
    inset_ax2.plot(range(len(w_dists) - len(w_dist_subset), len(w_dists)), w_dist_subset, color='green', linewidth=1)
    inset_ax2.set_title("W-Dist (Last 50)", fontsize=8)
    inset_ax2.set_xlabel("Epoch", fontsize=6)
    inset_ax2.set_ylabel("W-Dist", fontsize=6)
    inset_ax2.tick_params(axis='both', which='major', labelsize=5)
    inset_ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"{dataset_name.capitalize()} - {title}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()

# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, scalers = generate_datasets(n_samples=1000)
    
    configs = [
        {"hidden_dims": [128, 128], "name": "Shallow_(128,_128)"},
        {"hidden_dims": [512, 256, 128], "name": "Deep_(512,_256,_128)"},
    ]
    
    for dataset_name in ["moons", "blobs", "spirals"]:
        data = datasets[dataset_name]
        for config in configs:
            print(f"\nTraining on {dataset_name} with {config['name']}")
            model = MADE(input_dim=2, hidden_dims=config["hidden_dims"], output_dim=18, n_components=3)
            losses = train_made(model, data, epochs=300, device=device)
            
            # Track W-Dist over epochs
            w_dists = []
            data_orig = scalers[dataset_name].inverse_transform(data)
            for epoch in range(len(losses)):
                generated = sample_made(model, n_samples=1000, device=device, temperature_start=0.8, temperature_end=1.2)
                generated_orig = scalers[dataset_name].inverse_transform(generated)
                w_dist = compute_wasserstein_distance(data_orig, generated_orig)
                w_dists.append(w_dist)
            
            # Final sampling and visualization
            generated = sample_made(model, n_samples=1000, device=device, temperature_start=0.8, temperature_end=1.2)
            generated_orig = scalers[dataset_name].inverse_transform(generated)
            mean_dist = compute_mean_distance(data_orig, generated_orig)
            print(f"{dataset_name.capitalize()} Mean Distance (Original vs Generated): {mean_dist:.4f}")
            print(f"{dataset_name.capitalize()} Original points (first 5):\n", data_orig[:5])
            print(f"{dataset_name.capitalize()} Generated points (first 5):\n", generated_orig[:5])
            
            plot_results(
                data,
                generated,
                dataset_name,
                config["name"],
                f"{dataset_name}_{config['name']}.png",
                scalers[dataset_name],
                datasets,
                losses,
                w_dists
            )
            
            plt.figure()
            plt.plot(losses)
            plt.title(f"Training Loss - {dataset_name} {config['name']}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(f"loss_{dataset_name}_{config['name']}.png")
            plt.close()

if __name__ == "__main__":
    main()