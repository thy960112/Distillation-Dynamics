import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import timm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import argparse

def setup_device():
    """Set up computing device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def val_transform(input_size=224):
    """Create validation dataset preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize(int(input_size + 32)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def load_validation_data(val_dir, batch_size=32):
    """Load validation dataset"""
    transform = val_transform()
    
    val_dataset = ImageFolder(
        root=val_dir,
        transform=transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True  
    )
    
    return val_loader

def load_teacher_model_and_extract_features(model_name, device, images):
    """Load pretrained teacher model and extract intermediate features"""
    # Create and load pretrained model
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Extract intermediate features
    with torch.no_grad():
        _, block_outs = model.forward_intermediates(images)
    
    print(f"Teacher Model: {model_name}")
    print(f"Block outputs list length: {len(block_outs)}")
    if len(block_outs) > 0:
        print(f"Block output[0] shape: {block_outs[0].shape}")
    
    # Convert to numpy array [L, B, C, H, W]
    block_outs_cpu = [tensor.cpu() for tensor in block_outs]
    concatenated_tensor = torch.stack(block_outs_cpu, dim=0)
    return concatenated_tensor.numpy()

# FFT Analysis Functions
def perform_fft_on_channels(data):
    """
    Perform FFT on the C dimension
    
    Args:
    - data (np.ndarray): Input array with shape [L, B, C, H, W]
    
    Returns:
    - fft_data (np.ndarray): FFT result with shape [L, B, C, H, W]
    """
    L, B, C, H, W = data.shape
    fft_data = np.fft.fft(data, axis=2) / C
    return fft_data

def average_over_hw(fft_data):
    """
    Average over B, H and W dimensions
    
    Args:
    - fft_data (np.ndarray): FFT data with shape [L, B, C, H, W]
    
    Returns:
    - avg_freq_spectrum (np.ndarray): Averaged frequency spectrum with shape [L, C]
    """
    avg_freq_spectrum = np.mean(np.abs(fft_data), axis=(1, 3, 4))
    return avg_freq_spectrum

def average_over_channels(avg_freq_spectrum):
    """
    Average frequency intensity over C dimension
    
    Args:
    - avg_freq_spectrum (np.ndarray): Frequency spectrum with shape [L, C]
    
    Returns:
    - avg_freq_intensity (np.ndarray): Average frequency intensity with shape [L]
    """
    avg_freq_intensity = np.mean(avg_freq_spectrum, axis=1)
    return avg_freq_intensity

# Entropy Calculation Functions
def calculate_entropy_on_channels(data, num_bins=100):
    """
    Calculate Shannon entropy along the C dimension using binning
    
    Args:
    - data (np.ndarray): Input array with shape [L, B, C, H, W]
    - num_bins (int): Number of bins for discretization
    
    Returns:
    - entropy_data (np.ndarray): Entropy values with shape [L, B, H, W]
    """
    L, B, C, H, W = data.shape
    entropy_data = np.zeros((L, B, H, W))
    
    data_min = np.min(data)
    data_max = np.max(data)
    
    for l in range(L):
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    vector = data[l, b, :, h, w]
                    hist, _ = np.histogram(vector, bins=num_bins, range=(data_min, data_max))
                    probabilities = hist / np.sum(hist)
                    probabilities = probabilities[probabilities > 0]
                    
                    if len(probabilities) > 0:
                        entropy = -np.sum(probabilities * np.log2(probabilities))
                        entropy_data[l, b, h, w] = entropy
    
    return entropy_data

def average_over_bhw(entropy_data):
    """Average entropy values across B, H, and W dimensions"""
    avg_entropy = np.mean(entropy_data, axis=(1, 2, 3))
    return avg_entropy

# Plotting Functions
def plot_histogram(freq_intensity, bins=10, output_prefix=""):
    """Plot histogram of frequency intensity"""
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.hist(freq_intensity, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel('Average Frequency Intensity')
    plt.ylabel('Number of Layers')
    plt.title('Teacher Model: Frequency Intensity Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    filename = f"{output_prefix}_histogram.pdf"
    plt.savefig(filename, format='pdf')
    plt.close()
    print(f"Saved: {filename}")

def plot_bar_chart(freq_intensity, output_prefix=""):
    """Plot bar chart of frequency intensity"""
    N = len(freq_intensity)
    indices = np.arange(1, N + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = 'Times New Roman'
    ax.bar(indices, freq_intensity, color='skyblue', edgecolor='black')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Average Frequency Intensity')
    ax.set_title('Teacher Model: Frequency Intensity by Layer')
    ax.set_xticks(indices[::2])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    filename = f"{output_prefix}_frequency_intensity_bar.pdf"
    plt.savefig(filename, format='pdf')
    plt.close()
    print(f"Saved: {filename}")

def save_frequency_intensity_by_layer(positive_freqs, positive_magnitude, output_prefix=""):
    """Save frequency intensity plots for each layer"""
    num_layers = positive_magnitude.shape[0]
    
    for i in range(num_layers):
        plt.figure(figsize=(8, 6))
        plt.rcParams['font.size'] = 25
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.plot(positive_freqs, positive_magnitude[i, :], color='b', linewidth=1.5)
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Frequency Intensity')
        plt.title(f'Teacher Model - Layer {i+1}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = f"{output_prefix}_frequency_intensity_layer_{i}.pdf"
        plt.savefig(filename, format='pdf')
        plt.close()
    
    print(f"Saved {num_layers} frequency intensity plots for teacher model")

def plot_entropy_histogram(entropy_values, bins=10, output_prefix=""):
    """Plot histogram of entropy values"""
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.hist(entropy_values, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel('Shannon Entropy')
    plt.ylabel('Number of Layers')
    plt.title('Teacher Model: Entropy Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    filename = f"{output_prefix}_entropy_histogram.pdf"
    plt.savefig(filename, format='pdf')
    plt.close()
    print(f"Saved: {filename}")

def plot_entropy_bar_chart(entropy_values, output_prefix=""):
    """Plot bar chart of entropy values with log scale"""
    L = len(entropy_values)
    indices = np.arange(1, L + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plot_values = np.copy(entropy_values)
    plot_values[plot_values <= 0] = 1e-10
    
    ax.bar(indices, plot_values, color='skyblue', edgecolor='black')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Shannon Entropy (log scale)')
    ax.set_title('Teacher Model: Entropy by Layer')
    ax.set_xticks(indices[::2])
    ax.set_yscale('log')
    
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    filename = f"{output_prefix}_entropy_bar.pdf"
    plt.savefig(filename, format='pdf')
    plt.close()
    print(f"Saved: {filename}")

def save_entropy_data_to_csv(entropy_values, model_name, output_dir):
    """Save entropy data to CSV file"""
    import csv
    
    filename = os.path.join(output_dir, f"{model_name}_entropy_data.csv")
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Layer', 'Shannon_Entropy'])
        
        # Write data
        for i, entropy in enumerate(entropy_values):
            writer.writerow([i+1, entropy])
    
    print(f"Saved entropy data to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Teacher Model (CaiT) intermediate features')
    parser.add_argument('--val_dir', type=str, required=True, help='ImageNet validation directory')
    parser.add_argument('--output_dir', type=str, default='./teacher_analysis_output', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    parser.add_argument('--model_name', type=str, default='cait_s24_224', help='Teacher model name from timm')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup
    device = setup_device()
    
    # Load validation data
    print(f"Loading validation data from {args.val_dir}")
    val_loader = load_validation_data(args.val_dir, args.batch_size)
    images, _ = next(iter(val_loader))
    images = images.to(device)
    
    # Create output prefix
    output_prefix = os.path.join(args.output_dir, args.model_name)
    
    print(f"\nAnalyzing teacher model: {args.model_name}")
    
    # Extract features
    features = load_teacher_model_and_extract_features(args.model_name, device, images)
    
    # FFT Analysis
    print("\nPerforming FFT analysis...")
    fft_data = perform_fft_on_channels(features)
    avg_freq_spectrum = average_over_hw(fft_data)
    avg_freq_intensity = average_over_channels(avg_freq_spectrum)
    
    # Plot FFT results
    plot_histogram(avg_freq_intensity, bins=10, output_prefix=output_prefix)
    plot_bar_chart(avg_freq_intensity, output_prefix=output_prefix)
    
    # Frequency spectrum analysis
    C = avg_freq_spectrum.shape[-1]
    freqs = np.fft.fftfreq(C)
    positive_freqs = freqs[:C//2]
    positive_magnitude = avg_freq_spectrum[:, :C//2]
    
    save_frequency_intensity_by_layer(positive_freqs, positive_magnitude, output_prefix=output_prefix)
    
    # Entropy Analysis
    print("\nPerforming entropy analysis...")
    entropy_data = calculate_entropy_on_channels(features, num_bins=100)
    avg_entropy = average_over_bhw(entropy_data)
    
    # Plot entropy results
    plot_entropy_histogram(avg_entropy, bins=10, output_prefix=output_prefix)
    plot_entropy_bar_chart(avg_entropy, output_prefix=output_prefix)
    
    # Save entropy data
    save_entropy_data_to_csv(avg_entropy, args.model_name, args.output_dir)
    
    print(f"\nTeacher model analysis completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()