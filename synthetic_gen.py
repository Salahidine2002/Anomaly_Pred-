import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from scipy.signal import cwt as cwt_func
from scipy import signal as scipy_signal 
from pyts.image import MarkovTransitionField
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
import itertools

class SyntheticSignalDataset(Dataset):
    def __init__(self, n_samples, n_features=1, n_timesteps=100, 
                 image_size=24, transform_method='both', seed=None):
        """
        Dataset for generating synthetic time series signals on-the-fly.
        
        Args:
            n_samples (int): Number of samples in the dataset
            n_features (int): Number of features per series
            n_timesteps (int): Length of each series
            image_size (int): Size of output images
            transform_method (str): Transformation method ('mtf', 'gaf', or 'both')
            seed (int): Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.image_size = image_size
        self.transform_method = transform_method
        
        # Initialize transformers
        self.mtf = MarkovTransitionField(image_size=image_size, n_bins=6)
        self.gaf = GramianAngularField(image_size=image_size)
        self.scaler = MinMaxScaler()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
    
    def generate_signal(self):
        """Generate a single synthetic signal with diverse patterns."""
        t = np.linspace(0, 4*np.pi, self.n_timesteps)
        
        # Base signal components
        components = []
        
        # 1. Multiple frequency components with varying characteristics
        n_freqs = np.random.randint(1, 5)  # Increased max frequencies
        for _ in range(n_freqs):
            freq = np.random.uniform(0.05, 0.5)
            amp = np.random.uniform(0.3, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            # Add frequency modulation
            if np.random.random() < 0.3:
                mod_freq = np.random.uniform(0.01, 0.1)
                mod_amp = np.random.uniform(0.1, 0.5)
                freq = freq * (1 + mod_amp * np.sin(mod_freq * t))
            components.append(amp * np.sin(freq * t + phase))
        
        # 2. Trend components with more variety
        n_trends = np.random.randint(0, 3)  # Increased max trends
        for _ in range(n_trends):
            trend_type = np.random.choice(['linear', 'quadratic', 'exponential', 'logarithmic', 'sigmoid'])
            if trend_type == 'linear':
                components.append(np.random.uniform(-0.1, 0.1) * t)
            elif trend_type == 'quadratic':
                components.append(np.random.uniform(-0.01, 0.01) * t**2)
            elif trend_type == 'exponential':
                components.append(np.random.uniform(0.001, 0.01) * np.exp(t/100))
            elif trend_type == 'logarithmic':
                components.append(np.random.uniform(-0.1, 0.1) * np.log1p(t))
            else:  # sigmoid
                components.append(np.random.uniform(-0.5, 0.5) * (1 / (1 + np.exp(-t/50))))
        
        # 3. Seasonal components with varying periods and amplitudes
        n_seasons = np.random.randint(0, 3)  # Increased max seasons
        for _ in range(n_seasons):
            period = np.random.randint(20, 200)  # Increased period range
            amp = np.random.uniform(0.2, 1.0)
            phase = np.random.uniform(0, 2*np.pi)
            # Add amplitude modulation
            if np.random.random() < 0.3:
                mod_period = np.random.randint(100, 400)
                mod_amp = np.random.uniform(0.2, 0.8)
                amp = amp * (1 + mod_amp * np.sin(2*np.pi*t/mod_period))
            components.append(amp * np.sin(2*np.pi*t/period + phase))
        
        # 4. Random walk components with varying characteristics
        if np.random.random() < 0.7:  # Increased probability
            # Multiple random walks with different scales
            n_walks = np.random.randint(1, 3)
            for _ in range(n_walks):
                scale = np.random.uniform(0.05, 0.2)
                persistence = np.random.uniform(0.5, 0.9)
                walk = np.zeros(self.n_timesteps)
                walk[0] = np.random.normal(0, scale)
                for i in range(1, self.n_timesteps):
                    walk[i] = persistence * walk[i-1] + np.random.normal(0, scale)
                components.append(walk)
        
        # 5. Impulse components
        if np.random.random() < 0.5:
            n_impulses = np.random.randint(1, 4)
            for _ in range(n_impulses):
                pos = np.random.randint(0, self.n_timesteps)
                width = np.random.randint(2, 10)
                amplitude = np.random.uniform(0.5, 2.0)
                impulse = amplitude * np.exp(-(np.arange(self.n_timesteps) - pos)**2 / (2 * width**2))
                components.append(impulse)
        
        # 6. Step components
        if np.random.random() < 0.4:
            n_steps = np.random.randint(1, 3)
            for _ in range(n_steps):
                pos = np.random.randint(0, self.n_timesteps)
                amplitude = np.random.uniform(-1.0, 1.0)
                step = np.zeros(self.n_timesteps)
                step[pos:] = amplitude
                components.append(step)
        
        # Combine all components
        signal = np.sum(components, axis=0)
        
        # Add complex noise patterns
        noise_components = []
        
        # 1. White noise with varying intensity
        if np.random.random() < 0.8:
            noise_level = np.random.uniform(0.01, 0.1)
            noise_components.append(np.random.normal(0, noise_level, self.n_timesteps))
        
        # 2. Colored noise (1/f noise)
        if np.random.random() < 0.6:
            f = np.fft.fftfreq(self.n_timesteps)
            f[0] = 1e-6  # Avoid division by zero
            S = 1 / np.abs(f)
            noise = np.fft.ifft(np.random.normal(0, 1, self.n_timesteps) * np.sqrt(S))
            noise = noise.real
            noise = noise / np.std(noise) * np.random.uniform(0.01, 0.05)
            noise_components.append(noise)
        
        # 3. Localized noise bursts
        if np.random.random() < 0.5:
            n_bursts = np.random.randint(1, 4)
            for _ in range(n_bursts):
                pos = np.random.randint(0, self.n_timesteps)
                width = np.random.randint(5, 20)
                intensity = np.random.uniform(0.05, 0.2)
                burst = intensity * np.exp(-(np.arange(self.n_timesteps) - pos)**2 / (2 * width**2))
                burst *= np.random.normal(0, 1, self.n_timesteps)
                noise_components.append(burst)
        
        # 4. Non-Gaussian noise
        if np.random.random() < 0.4:
            noise_type = np.random.choice(['laplace', 'cauchy', 'student_t'])
            if noise_type == 'laplace':
                noise = np.random.laplace(0, np.random.uniform(0.01, 0.05), self.n_timesteps)
            elif noise_type == 'cauchy':
                noise = np.random.standard_cauchy(self.n_timesteps) * np.random.uniform(0.01, 0.05)
            else:  # student_t
                df = np.random.uniform(1, 5)
                noise = np.random.standard_t(df, self.n_timesteps) * np.random.uniform(0.01, 0.05)
            noise_components.append(noise)
        
        # Combine all noise components
        noise = np.sum(noise_components, axis=0)
        signal += noise
        

        signal = 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
        
        return signal
    
    def transform_to_image(self, signal):
        """Transform a single signal to image representation using multiple methods."""
        
        # Reshape for transformation
        signal_scaled = signal.reshape(1, -1)
        
        # Initialize list to store all transformations
        transformations = []
        
        if self.transform_method == 'all':
            # 1. Markov Transition Field (MTF)
            mtf = MarkovTransitionField(image_size=self.image_size, n_bins=10)
            transformations.append(mtf.fit_transform(signal_scaled)[0])
            
            # 2. Gramian Angular Field (GAF)
            gasf = GramianAngularField(image_size=self.image_size, method='summation')
            transformations.append(gasf.fit_transform(signal_scaled)[0])

            gadf = GramianAngularField(image_size=self.image_size, method='difference')
            transformations.append(gadf.fit_transform(signal_scaled)[0])
            
            # 3. Recurrence Plot (RP)
            # Create recurrence plot using scipy's correlation
            rp = np.zeros((self.image_size, self.image_size))
            for i in range(self.image_size):
                for j in range(self.image_size):
                    idx_i = int(i * len(signal_scaled[0]) / self.image_size)
                    idx_j = int(j * len(signal_scaled[0]) / self.image_size)
                    rp[i, j] = np.abs(signal_scaled[0, idx_i] - signal_scaled[0, idx_j])
            transformations.append(rp.reshape(self.image_size, self.image_size))
            
            # 4. Continuous Wavelet Transform (CWT)
            # Using Morlet wavelet
            # scales = np.linspace(1, self.image_size, self.image_size)
            # cwt = cwt_func(signal_scaled[0], scipy_signal.morlet2, scales)
            # cwt = np.abs(cwt)
            # Resize to match image size
            # cwt = scipy_signal.resample(cwt, self.image_size, axis=0)
            # cwt = scipy_signal.resample(cwt, self.image_size, axis=1)
            # transformations.append(cwt.reshape(self.image_size, self.image_size))

            # Combine all transformations
            image = np.array(transformations)
        
        # elif self.transform_method == 'mtf':
        #     mtf = MarkovTransitionField(image_size=self.image_size, n_bins=10)
        #     image = mtf.fit_transform(signal_scaled)
        
        # elif self.transform_method == 'gaf':
        #     gaf = GramianAngularField(image_size=self.image_size, method='summation')
        #     image = gaf.fit_transform(signal_scaled)
        
        # else:  # both (default)
        #     mtf = MarkovTransitionField(image_size=self.image_size, n_bins=10)
        #     gaf = GramianAngularField(image_size=self.image_size, method='summation')
            
        #     image_mtf = mtf.fit_transform(signal_scaled)
        #     image_gaf = gaf.fit_transform(signal_scaled)
            
        #     image = np.array([image_mtf, image_gaf], axis=1)
        
        # Normalize the final image to [0, 1] range for better visualization
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        return image
    
    def _permutation_entropy(self, x):
        """Calculate permutation entropy for a window of data."""
        if len(x) < 3:
            return 0
        # Get all possible permutations
        perms = np.array(list(itertools.permutations(range(3))))
        # Count occurrences of each permutation
        counts = np.zeros(len(perms))
        for i in range(len(x)-2):
            window = x[i:i+3]
            rank = np.argsort(window)
            for j, perm in enumerate(perms):
                if np.array_equal(rank, perm):
                    counts[j] += 1
                    break
        # Calculate entropy
        p = counts / np.sum(counts)
        p = p[p > 0]  # Remove zero probabilities
        return -np.sum(p * np.log2(p))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Generate signal
        signal = self.generate_signal()
        
        # Transform to image
        image = self.transform_to_image(signal)

        sigma = np.std(signal)
        if not sigma: 
            sigma = 1
        signal = (signal-np.mean(signal))/sigma
        
        # Convert to tensors
        signal = torch.FloatTensor(signal)
        image = torch.FloatTensor(image)
        
        return {
            'signal': signal,
            'image': image
        }

def get_dataloader(n_samples, batch_size=32, n_features=1, n_timesteps=100,
                  image_size=24, transform_method='both', seed=None,
                  num_workers=4, shuffle=True):
    """
    Create a DataLoader for synthetic signals.
    
    Args:
        n_samples (int): Number of samples in the dataset
        batch_size (int): Batch size for the DataLoader
        n_features (int): Number of features per series
        n_timesteps (int): Length of each series
        image_size (int): Size of output images
        transform_method (str): Transformation method ('mtf', 'gaf', or 'both')
        seed (int): Random seed for reproducibility
        num_workers (int): Number of worker processes
        shuffle (bool): Whether to shuffle the data
    
    Returns:
        DataLoader: PyTorch DataLoader for synthetic signals
    """
    dataset = SyntheticSignalDataset(
        n_samples=n_samples,
        n_features=n_features,
        n_timesteps=n_timesteps,
        image_size=image_size,
        transform_method=transform_method,
        seed=seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def visualize_batch(batch, n_samples=4):
    """
    Visualize a batch of signals and their transformations.
    
    Args:
        batch (dict): Batch containing 'signal' and 'image' tensors
        n_samples (int): Number of samples to visualize
    """
    signals = batch['signal'].numpy()
    images = batch['image'].numpy()
    
    n_samples = min(n_samples, len(signals))
    n_cols = 2 if images.shape[-1] == 1 else 3
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(5*n_cols, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Plot original signal
        axes[i, 0].plot(signals[i])
        axes[i, 0].set_title('Original Signal')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Value')
        
        if n_cols == 2:
            # Plot single transformation
            axes[i, 1].imshow(images[i, :, :, 0], cmap='viridis')
            axes[i, 1].set_title('Transformed Image')
            axes[i, 1].set_xlabel('Time')
            axes[i, 1].set_ylabel('Time')
        else:
            # Plot both transformations
            axes[i, 1].imshow(images[i, :, :, 0], cmap='viridis')
            axes[i, 1].set_title('MTF Image')
            axes[i, 1].set_xlabel('Time')
            axes[i, 1].set_ylabel('Time')
            
            axes[i, 2].imshow(images[i, :, :, 1], cmap='viridis')
            axes[i, 2].set_title('GAF Image')
            axes[i, 2].set_xlabel('Time')
            axes[i, 2].set_ylabel('Time')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    dataloader = get_dataloader(
        n_samples=1000,
        batch_size=32,
        n_features=1,
        n_timesteps=100,
        image_size=24,
        transform_method='both'
    )
    
    # Test the dataloader and visualize results
    for batch in dataloader:
        print(f"Batch shapes:")
        print(f"Signals: {batch['signal'].shape}")
        print(f"Images: {batch['image'].shape}")
        
        # Visualize the first batch
        visualize_batch(batch, n_samples=4)
        break  # Just print the first batch
