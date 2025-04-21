import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import logging
from tqdm import tqdm
from synthetic_gen import get_dataloader
from torch.distributions import StudentT, MixtureSameFamily, Independent, Normal
from torch.cuda.amp import autocast, GradScaler

class VAE(nn.Module):
    def __init__(self, latent_dim=128, input_channels=4, image_size=64, signal_length=600, n_components=3):
        super(VAE, self).__init__()
        
        self.image_size = image_size
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.n_components = n_components
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Calculate the size of flattened features from encoder
        self.flattened_size = 512 * (image_size // 16) * (image_size // 16)
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)
        
        # Enhanced decoder for mixture model parameters
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
        )
        
        # Output layers for mixture model parameters
        # Each component needs: mixture weight, location, scale, and degrees of freedom
        self.mixture_weights = nn.Linear(2048, n_components)
        self.locations = nn.Linear(2048, n_components * signal_length)
        self.scales = nn.Linear(2048, n_components * signal_length)
        self.dofs = nn.Linear(2048, n_components)
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # Get base features
        x = self.decoder(z)
        
        # Get mixture model parameters
        # Mixture weights (π)
        logits = self.mixture_weights(x)
        mixture_weights = F.softmax(logits, dim=-1)
        
        # Locations (μ)
        locations = self.locations(x).view(-1, self.n_components, self.signal_length)
        
        # Scales (σ) - ensure positive
        scales = F.softplus(self.scales(x)).view(-1, self.n_components, self.signal_length)
        
        # Degrees of freedom (ν) - ensure > 2
        dofs = F.softplus(self.dofs(x)) + 2.0
        
        return mixture_weights, locations, scales, dofs
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        mixture_weights, locations, scales, dofs = self.decode(z)
        return mixture_weights, locations, scales, dofs, mu, log_var, z
    
    def create_mixture_distribution(self, mixture_weights, locations, scales, dofs):
        """
        Creates a mixture of Student's t distributions from the given parameters.
        
        Args:
            mixture_weights: mixture component weights [batch_size, n_components]
            locations: location parameters for each component [batch_size, n_components, signal_length]
            scales: scale parameters for each component [batch_size, n_components, signal_length]
            dofs: degrees of freedom for each component [batch_size, n_components]
            
        Returns:
            MixtureSameFamily distribution
        """
        # Create mixture distribution
        mixture = MixtureSameFamily(
            torch.distributions.Categorical(mixture_weights),
            torch.distributions.Independent(
                torch.distributions.StudentT(
                    df=dofs.unsqueeze(-1).expand(-1, -1, self.signal_length),
                    loc=locations,
                    scale=scales
                ),
                1
            )
        )
        return mixture

    def sample_from_mixture(self, mixture, num_samples=1):
        """
        Samples from a mixture distribution.
        
        Args:
            mixture: MixtureSameFamily distribution
            num_samples: number of samples to generate
            
        Returns:
            samples from the mixture distribution
        """
        return mixture.sample((num_samples,))

    def loss_function(self, mixture_weights, locations, scales, dofs, target, mu, log_var, beta=1.0):
        """
        Computes the VAE loss function using Student's t mixture model.
        
        Args:
            mixture_weights: mixture component weights
            locations: location parameters for each component
            scales: scale parameters for each component
            dofs: degrees of freedom for each component
            target: target signal
            mu: mean of latent distribution
            log_var: log variance of latent distribution
            beta: weight for KL divergence term
            
        Returns:
            (loss, reconstruction_loss, KLD)
        """
        # Create mixture distribution
        mixture = self.create_mixture_distribution(mixture_weights, locations, scales, dofs)
        
        # Compute negative log likelihood
        nll = -mixture.log_prob(target)
        reconstruction_loss = nll.mean()
        
        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        loss = reconstruction_loss + beta * KLD
        return loss, reconstruction_loss, KLD
    
    def predict_signals(self, batch, num_samples=1, return_mean=False):
        """
        Takes a batch of inputs, computes the output distribution, and returns either samples or mean.
        
        Args:
            batch: dictionary containing 'image' tensor
            num_samples: number of samples to generate per input (ignored if return_mean=True)
            return_mean: if True, returns the distribution mean instead of samples
            
        Returns:
            if return_mean:
                mean_signals: tensor of shape [batch_size, signal_length]
                mixture_params: tuple containing (mixture_weights, locations, scales, dofs)
            else:
                sampled_signals: tensor of shape [batch_size, num_samples, signal_length]
                mixture_params: tuple containing (mixture_weights, locations, scales, dofs)
        """
        with torch.no_grad():
            # Get model outputs
            mixture_weights, locations, scales, dofs, mu, log_var, z = self(batch['image'])
            
            if return_mean:
                # Compute weighted mean across components
                # mixture_weights: [batch_size, n_components]
                # locations: [batch_size, n_components, signal_length]
                mean_signals = torch.sum(mixture_weights.unsqueeze(-1) * locations, dim=1)
                return mean_signals
            else:
                # Create mixture distribution
                mixture = self.create_mixture_distribution(mixture_weights, locations, scales, dofs)
                
                # Sample from the mixture
                samples = self.sample_from_mixture(mixture, num_samples)
                
                # Reshape samples to [batch_size, num_samples, signal_length]
                samples = samples.view(-1, num_samples, self.signal_length)
                
                return samples

    def generate(self, num_samples=1, device='cuda'):
        """
        Generate signals from random latent vectors using the mixture model.
        
        Args:
            num_samples: number of samples to generate
            device: device to generate samples on
            
        Returns:
            generated signals
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            mixture_weights, locations, scales, dofs = self.decode(z)
            
            # Create mixture distribution
            mixture = self.create_mixture_distribution(mixture_weights, locations, scales, dofs)
            
            # Sample from the mixture
            samples = self.sample_from_mixture(mixture)
        return samples

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device='cuda'):
        """
        Load a VAE model from a saved checkpoint.
        
        Args:
            checkpoint_path: path to the saved checkpoint file
            device: device to load the model on
            
        Returns:
            loaded model and optimizer
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model parameters from checkpoint
        model_params = {
            'latent_dim': checkpoint.get('latent_dim', 128),
            'input_channels': checkpoint.get('input_channels', 4),
            'image_size': checkpoint.get('image_size', 64),
            'signal_length': checkpoint.get('signal_length', 600),
            'n_components': checkpoint.get('n_components', 3)
        }
        
        # Create model instance
        model = cls(**model_params)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Create and load optimizer state if available
        optimizer = None
        if 'optimizer_state_dict' in checkpoint:
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Log loading information
        logging.info(f'Loaded model from checkpoint: {checkpoint_path}')
        logging.info(f'Model parameters: {model_params}')
        if optimizer:
            logging.info('Optimizer state loaded')
        
        return model, optimizer

class VAE1D(nn.Module):
    def __init__(self, latent_dim=128, signal_length=600, n_components=3):
        super(VAE1D, self).__init__()
        
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.n_components = n_components
        
        # 1D Encoder
        self.encoder = nn.Sequential(
            # Layer 1: signal_length -> signal_length/4
            nn.Conv1d(1, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Layer 2: signal_length/4 -> signal_length/16
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Layer 3: signal_length/16 -> signal_length/64
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Layer 4: signal_length/64 -> signal_length/256
            nn.Conv1d(256, 512, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Calculate the size of flattened features from encoder
        self.flattened_size = 512 * (signal_length // 256)  # Due to 4 layers of stride 4
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)
        
        # Enhanced decoder for mixture model parameters (same as original VAE)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
        )
        
        # Output layers for mixture model parameters (same as original VAE)
        self.mixture_weights = nn.Linear(2048, n_components)
        self.locations = nn.Linear(2048, n_components * signal_length)
        self.scales = nn.Linear(2048, n_components * signal_length)
        self.dofs = nn.Linear(2048, n_components)
        
    def encode(self, x):
        # Reshape input to [batch_size, channels=1, signal_length]
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # Same as original VAE
        x = self.decoder(z)
        
        # Get mixture model parameters
        logits = self.mixture_weights(x)
        mixture_weights = F.softmax(logits, dim=-1)
        
        locations = self.locations(x).view(-1, self.n_components, self.signal_length)
        scales = F.softplus(self.scales(x)).view(-1, self.n_components, self.signal_length)
        dofs = F.softplus(self.dofs(x)) + 2.0
        
        return mixture_weights, locations, scales, dofs
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        mixture_weights, locations, scales, dofs = self.decode(z)
        return mixture_weights, locations, scales, dofs, mu, log_var, z
    
    # Reuse the same distribution-related methods from original VAE
    create_mixture_distribution = VAE.create_mixture_distribution
    sample_from_mixture = VAE.sample_from_mixture
    loss_function = VAE.loss_function
    
    def predict_signals(self, batch, num_samples=1, return_mean=False):
        """
        Takes a batch of inputs, computes the output distribution, and returns either samples or mean.
        
        Args:
            batch: dictionary containing 'signal' tensor
            num_samples: number of samples to generate per input (ignored if return_mean=True)
            return_mean: if True, returns the distribution mean instead of samples
            
        Returns:
            if return_mean:
                mean_signals: tensor of shape [batch_size, signal_length]
            else:
                sampled_signals: tensor of shape [batch_size, num_samples, signal_length]
        """
        with torch.no_grad():
            # Get model outputs
            mixture_weights, locations, scales, dofs, mu, log_var, z = self(batch['signal'])
            
            if return_mean:
                # Compute weighted mean across components
                mean_signals = torch.sum(mixture_weights.unsqueeze(-1) * locations, dim=1)
                return mean_signals
            else:
                # Create mixture distribution
                mixture = self.create_mixture_distribution(mixture_weights, locations, scales, dofs)
                
                # Sample from the mixture
                samples = self.sample_from_mixture(mixture, num_samples)
                
                # Reshape samples to [batch_size, num_samples, signal_length]
                samples = samples.view(-1, num_samples, self.signal_length)
                
                return samples
    
    def generate(self, num_samples=1, device='cuda'):
        """
        Generate signals from random latent vectors using the mixture model.
        
        Args:
            num_samples: number of samples to generate
            device: device to generate samples on
            
        Returns:
            generated signals
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            mixture_weights, locations, scales, dofs = self.decode(z)
            
            # Create mixture distribution
            mixture = self.create_mixture_distribution(mixture_weights, locations, scales, dofs)
            
            # Sample from the mixture
            samples = self.sample_from_mixture(mixture)
        return samples
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device='cuda'):
        """
        Load a VAE1D model from a saved checkpoint.
        
        Args:
            checkpoint_path: path to the saved checkpoint file
            device: device to load the model on
            
        Returns:
            loaded model and optimizer
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model parameters from checkpoint
        model_params = {
            'latent_dim': checkpoint.get('latent_dim', 128),
            'signal_length': checkpoint.get('signal_length', 600),
            'n_components': checkpoint.get('n_components', 3)
        }
        
        # Create model instance
        model = cls(**model_params)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Create and load optimizer state if available
        optimizer = None
        if 'optimizer_state_dict' in checkpoint:
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Log loading information
        logging.info(f'Loaded model from checkpoint: {checkpoint_path}')
        logging.info(f'Model parameters: {model_params}')
        if optimizer:
            logging.info('Optimizer state loaded')
        
        return model, optimizer

def setup_logging(log_dir):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, log_dir, beta=1.0):
    """
    Train the VAE model for signal reconstruction with mixed precision

    Args:
        model: VAE or VAE1D model instance
        train_loader: training data loader
        val_loader: validation data loader
        optimizer: optimizer instance
        num_epochs: number of epochs to train
        device: device to train on
        log_dir: directory to save logs and checkpoints
        beta: weight for KL divergence term
    """
    model = model.to(device)
    best_val_loss = float('inf')
    scaler = GradScaler()  # For mixed precision

    # Determine model type
    is_1d_model = isinstance(model, VAE1D)
    input_key = 'signal' if is_1d_model else 'image'
    model_type = 'VAE1D' if is_1d_model else 'VAE'

    # Create loss log file
    loss_log_file = os.path.join(log_dir, 'losses.txt')
    logging.info(f'Training {model_type} model')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_signal_loss = 0
        train_kld_loss = 0

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                data = batch[input_key].to(device, non_blocking=True)
                signals = batch['signal'].to(device, non_blocking=True)

                optimizer.zero_grad()
                with autocast():  # Mixed precision context
                    mixture_weights, locations, scales, dofs, mu, log_var, z = model(data)
                    loss, signal_loss, kld_loss = model.loss_function(
                        mixture_weights, locations, scales, dofs, signals, mu, log_var, beta
                    )

                scaler.scale(loss).backward()       # Scaled backward
                scaler.step(optimizer)              # Scaled optimizer step
                scaler.update()                     # Update the scale for next iteration

                train_loss += loss.item()
                train_signal_loss += signal_loss.item()
                train_kld_loss += kld_loss.item()

                pbar.set_postfix({
                    'loss': loss.item() / len(data),
                    'signal_loss': signal_loss.item() / len(data),
                    'kld_loss': kld_loss.item() / len(data)
                })

        # Validation phase (no autocast here — inference is safe in FP32)
        model.eval()
        val_loss = 0
        val_signal_loss = 0
        val_kld_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[input_key].to(device)
                signals = batch['signal'].to(device)

                mixture_weights, locations, scales, dofs, mu, log_var, z = model(data)
                loss, signal_loss, kld_loss = model.loss_function(
                    mixture_weights, locations, scales, dofs, signals, mu, log_var, beta
                )

                val_loss += loss.item()
                val_signal_loss += signal_loss.item()
                val_kld_loss += kld_loss.item()

        # Average losses
        train_loss /= len(train_loader.dataset)
        train_signal_loss /= len(train_loader.dataset)
        train_kld_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_signal_loss /= len(val_loader.dataset)
        val_kld_loss /= len(val_loader.dataset)

        # Log losses
        with open(loss_log_file, 'a') as f:
            f.write(f'Epoch {epoch+1}:\n')
            f.write(f'Train Loss: {train_loss:.6f}, Train Signal Loss: {train_signal_loss:.6f}, Train KLD: {train_kld_loss:.6f}\n')
            f.write(f'Val Loss: {val_loss:.6f}, Val Signal Loss: {val_signal_loss:.6f}, Val KLD: {val_kld_loss:.6f}\n\n')

        logging.info(f'Epoch {epoch+1}:')
        logging.info(f'Train Loss: {train_loss:.6f}, Train Signal Loss: {train_signal_loss:.6f}, Train KLD Loss: {train_kld_loss:.6f}')
        logging.info(f'Val Loss: {val_loss:.6f}, Val Signal Loss: {val_signal_loss:.6f}, Val KLD Loss: {val_kld_loss:.6f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_params = {
                'latent_dim': model.latent_dim,
                'signal_length': model.signal_length,
                'n_components': model.n_components
            }
            if not is_1d_model:
                model_params.update({
                    'input_channels': model.encoder[0].in_channels,
                    'image_size': model.image_size
                })

            torch.save({
                'model_state_dict': model.state_dict(),
                **model_params
            }, os.path.join(log_dir, 'best_model.pth'))
            logging.info('Saved best model')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_type': model_type,
            **model_params
        }, os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        logging.info(f'Saved checkpoint for epoch {epoch+1}')

        return model


def main():
    parser = argparse.ArgumentParser(description='Train VAE model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for KL divergence term')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of synthetic samples to generate')
    parser.add_argument('--n_features', type=int, default=1, help='Number of features per series')
    parser.add_argument('--n_timesteps', type=int, default=600, help='Length of each series')
    parser.add_argument('--image_size', type=int, default=64, help='Size of output images')
    parser.add_argument('--transform_method', type=str, default='both', help='Transformation method (mtf, gaf, or both)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(output_dir)
    logging.info(f'Starting training with arguments: {args}')
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Generate synthetic data
    logging.info('Generating synthetic data...')
    
    # Split data
    train_size = int(0.8 * args.n_samples)
    val_size = args.n_samples - train_size
    
    # Create dataloaders
    train_loader = get_dataloader(
        n_samples=train_size,
        batch_size=args.batch_size,
        n_features=args.n_features,
        n_timesteps=args.n_timesteps,
        image_size=args.image_size,
        transform_method=args.transform_method,
        seed=args.seed,
        num_workers=4,
        shuffle=True
    )
    
    val_loader = get_dataloader(
        n_samples=val_size,
        batch_size=args.batch_size,
        n_features=args.n_features,
        n_timesteps=args.n_timesteps,
        image_size=args.image_size,
        transform_method=args.transform_method,
        seed=args.seed + 1,
        num_workers=4,
        shuffle=False
    )
    
    logging.info(f'Generated {train_size} training samples and {val_size} validation samples')
    
    # Initialize model
    model = VAE(
        latent_dim=args.latent_dim,
        input_channels=4,
        image_size=args.image_size,
        signal_length=args.n_timesteps
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        log_dir=output_dir,
        beta=args.beta
    )
    
    logging.info('Training completed')

if __name__ == '__main__':
    main()

