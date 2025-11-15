import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import timm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import cv2
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class GradientAnalyzer:
    """Extract and analyze gradients through CaiT model layers"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = {}
        self.gradients = {}
        self.handles = []
        
    def _register_hooks(self, target_layers: Optional[List[str]] = None):
        """Register forward and backward hooks on specified layers"""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # If no specific layers provided, register on all blocks
        if target_layers is None:
            target_layers = []
            
            # Register hooks for patch self-attention blocks
            if hasattr(self.model, 'blocks'):
                for idx, block in enumerate(self.model.blocks):
                    target_layers.append(f'blocks.{idx}')
            
            # Register hooks for class-attention blocks
            if hasattr(self.model, 'blocks_token_only'):
                for idx, block in enumerate(self.model.blocks_token_only):
                    target_layers.append(f'blocks_token_only.{idx}')
        
        # Register the hooks
        for layer_name in target_layers:
            parts = layer_name.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
            
            # Register both forward and backward hooks
            handle_fwd = module.register_forward_hook(get_activation(layer_name))
            handle_bwd = module.register_backward_hook(get_gradient(layer_name))
            self.handles.extend([handle_fwd, handle_bwd])
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def compute_saliency(self, image, target_class=None, method='vanilla'):
        """Compute saliency map using specified method"""
        self.model.eval()
        
        # Ensure image is a leaf variable that requires grad
        image = image.detach().clone()
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        
        # Get target class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Select the target class score
        if isinstance(target_class, int):
            target_score = output[0, target_class]
        else:
            # Ensure target_class is on the same device as output
            target_class = target_class.to(output.device)
            target_score = output.gather(1, target_class.view(-1, 1)).squeeze()
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Get gradients with respect to input
        saliency = image.grad.data.abs()
        
        if method == 'vanilla':
            # Standard gradient
            saliency = saliency
        elif method == 'smooth':
            # SmoothGrad: average gradients with noise
            saliency = self._smooth_grad(image, target_class, n_samples=50, noise_level=0.15)
        elif method == 'integrated':
            # Integrated gradients
            saliency = self._integrated_gradients(image, target_class, steps=50)
        
        return saliency
    
    def _smooth_grad(self, image, target_class, n_samples=50, noise_level=0.15):
        """Compute SmoothGrad saliency"""
        self.model.eval()
        
        # Ensure image is detached
        image = image.detach()
        
        # Compute stdev of noise
        stdev = noise_level * (image.max() - image.min())
        
        total_gradients = torch.zeros_like(image)
        
        for _ in range(n_samples):
            # Add noise to input
            noise = torch.randn_like(image) * stdev
            noisy_image = (image + noise).clone()
            noisy_image.requires_grad = True
            
            # Forward and backward
            output = self.model(noisy_image)
            
            if isinstance(target_class, int):
                target_score = output[0, target_class]
            else:
                # Ensure target_class is on the same device as output
                target_class = target_class.to(output.device)
                target_score = output.gather(1, target_class.view(-1, 1)).squeeze()
            
            self.model.zero_grad()
            target_score.backward()
            
            # Accumulate gradients
            total_gradients += noisy_image.grad.data.abs()
        
        return total_gradients / n_samples
    
    def _integrated_gradients(self, image, target_class, steps=50):
        """Compute Integrated Gradients"""
        self.model.eval()
        
        # Ensure image is detached
        image = image.detach()
        
        # Create baseline (black image)
        baseline = torch.zeros_like(image)
        
        # Generate alphas
        alphas = torch.linspace(0, 1, steps + 1).to(self.device)
        
        # Initialize integrated gradients
        integrated_grads = torch.zeros_like(image)
        
        for i in range(steps):
            # Interpolate between baseline and input
            interpolated = (baseline + alphas[i] * (image - baseline)).clone()
            interpolated.requires_grad = True
            
            # Forward pass
            output = self.model(interpolated)
            
            if isinstance(target_class, int):
                target_score = output[0, target_class]
            else:
                # Ensure target_class is on the same device as output
                target_class = target_class.to(output.device)
                target_score = output.gather(1, target_class.view(-1, 1)).squeeze()
            
            # Backward pass
            self.model.zero_grad()
            target_score.backward()
            
            # Accumulate gradients
            integrated_grads += interpolated.grad.data
        
        # Average and scale
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (image - baseline)
        
        return integrated_grads.abs()
    
    def analyze_layer_gradients(self, image, target_class=None):
        """Analyze how gradients flow through different layers"""
        self.activations = {}
        self.gradients = {}
        self._register_hooks()
        
        self.model.eval()
        
        # Ensure image is a leaf variable that requires grad
        image = image.detach().clone()
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Select target score
        if isinstance(target_class, int):
            target_score = output[0, target_class]
        else:
            # Ensure target_class is on the same device as output
            target_class = target_class.to(output.device)
            target_score = output.gather(1, target_class.view(-1, 1)).squeeze()
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Store input gradient
        input_gradient = image.grad.data.clone()
        
        self._remove_hooks()
        
        return {
            'input_gradient': input_gradient,
            'activations': self.activations,
            'gradients': self.gradients,
            'output': output,
            'target_class': target_class
        }

class SaliencyVisualizer:
    """Visualize saliency maps and gradient flow"""
    
    @staticmethod
    def process_saliency(saliency_map):
        """Process saliency map for visualization"""
        # Take the maximum across color channels
        saliency = saliency_map.squeeze(0).cpu()
        if saliency.dim() == 3:
            saliency = saliency.max(dim=0)[0]
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency.numpy()
    
    @staticmethod
    def visualize_saliency_comparison(image_np, saliency_maps, methods, output_path):
        """Compare different saliency methods"""
        n_methods = len(methods)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4*(n_methods+1), 8))
        
        # Original image
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Different saliency methods
        for idx, (method, saliency) in enumerate(zip(methods, saliency_maps)):
            processed_saliency = SaliencyVisualizer.process_saliency(saliency)
            
            # Top row: saliency map
            im = axes[0, idx+1].imshow(processed_saliency, cmap='hot')
            axes[0, idx+1].set_title(f'{method.capitalize()} Gradient', fontsize=14)
            axes[0, idx+1].axis('off')
            
            # Bottom row: overlay
            axes[1, idx+1].imshow(image_np)
            axes[1, idx+1].imshow(processed_saliency, cmap='hot', alpha=0.5)
            axes[1, idx+1].set_title('Overlay', fontsize=12)
            axes[1, idx+1].axis('off')
        
        plt.suptitle('Saliency Map Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def visualize_layer_gradients(layer_analysis, image_np, output_dir):
        """Visualize gradient flow through layers"""
        # Create directory for layer visualizations
        layer_dir = os.path.join(output_dir, 'layer_gradients')
        os.makedirs(layer_dir, exist_ok=True)
        
        # Process input gradient
        input_grad = SaliencyVisualizer.process_saliency(layer_analysis['input_gradient'])
        
        # Get layer names sorted by depth
        layer_names = sorted([k for k in layer_analysis['gradients'].keys() 
                            if 'blocks' in k], 
                           key=lambda x: (x.split('.')[0], int(x.split('.')[1])))
        
        # Visualize gradient magnitude evolution
        gradient_magnitudes = []
        activation_magnitudes = []
        
        for layer_name in layer_names:
            if layer_name in layer_analysis['gradients']:
                grad = layer_analysis['gradients'][layer_name]
                act = layer_analysis['activations'][layer_name]
                
                # Compute average magnitudes
                grad_mag = grad.abs().mean().item()
                act_mag = act.abs().mean().item()
                
                gradient_magnitudes.append(grad_mag)
                activation_magnitudes.append(act_mag)
        
        # Create gradient flow plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gradient magnitudes
        ax1.plot(range(24), gradient_magnitudes[:24], 'b-o', linewidth=2)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Average Gradient Magnitude')
        ax1.set_title('Gradient Magnitude Through Layers')
        ax1.grid(True, alpha=0.3)
        
        # Activation magnitudes
        ax2.plot(range(24), activation_magnitudes[:24], 'r-o', linewidth=2)
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Average Activation Magnitude')
        ax2.set_title('Activation Magnitude Through Layers')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Gradient and Activation Flow Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_flow.pdf'), dpi=150)
        plt.close()
        
        # Visualize gradients at different layers
        n_layers_to_show = min(5, len(layer_names))
        layer_indices = np.linspace(0, len(layer_names)-1, n_layers_to_show, dtype=int)
        
        fig, axes = plt.subplots(2, n_layers_to_show + 1, figsize=(4*(n_layers_to_show+1), 8))
        
        # Input gradient
        axes[0, 0].imshow(input_grad, cmap='hot')
        axes[0, 0].set_title('Input Gradient', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(image_np)
        axes[1, 0].imshow(input_grad, cmap='hot', alpha=0.5)
        axes[1, 0].set_title('Input Overlay', fontsize=12)
        axes[1, 0].axis('off')
        
        # Layer gradients
        for idx, layer_idx in enumerate(layer_indices):
            layer_name = layer_names[layer_idx]
            
            # Get gradient for this layer
            grad = layer_analysis['gradients'][layer_name]
            
            # For transformer blocks, we need to handle the sequence dimension
            if grad.dim() == 3:  # [B, N, D]
                # Take the gradient magnitude across the feature dimension
                grad_mag = grad.abs().mean(dim=-1)[0]  # [N]
                
                # If this includes CLS token, separate it
                if hasattr(grad_mag, 'shape') and grad_mag.shape[0] > 196:
                    cls_grad = grad_mag[0].item()
                    patch_grad = grad_mag[1:]
                else:
                    patch_grad = grad_mag
                
                # Reshape to 2D if possible
                n_patches = int(np.sqrt(len(patch_grad)))
                if n_patches * n_patches == len(patch_grad):
                    grad_2d = patch_grad.reshape(n_patches, n_patches).cpu().numpy()
                else:
                    # Handle non-square
                    side = int(np.sqrt(len(patch_grad)))
                    if side * side < len(patch_grad):
                        side += 1
                    padded = torch.zeros(side * side)
                    padded[:len(patch_grad)] = patch_grad
                    grad_2d = padded.reshape(side, side).cpu().numpy()
                
                # Visualize
                im = axes[0, idx+1].imshow(grad_2d, cmap='hot')
                axes[0, idx+1].set_title(f'Layer {layer_idx+1}\n{layer_name}', fontsize=10)
                axes[0, idx+1].axis('off')
                
                # Overlay
                grad_resized = cv2.resize(grad_2d, (224, 224), interpolation=cv2.INTER_CUBIC)
                grad_resized = (grad_resized - grad_resized.min()) / (grad_resized.max() - grad_resized.min() + 1e-8)
                
                axes[1, idx+1].imshow(image_np)
                axes[1, idx+1].imshow(grad_resized, cmap='hot', alpha=0.5)
                axes[1, idx+1].axis('off')
        
        plt.suptitle('Layer-wise Gradient Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_gradients.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
        return gradient_magnitudes, activation_magnitudes

def create_gradient_evolution_gif(analyzer, image, image_np, output_dir, target_class=None):
    """Create GIF showing gradient evolution through layers"""
    import imageio
    
    # Get layer analysis
    layer_analysis = analyzer.analyze_layer_gradients(image, target_class)
    
    # Create temporary directory for frames
    temp_dir = os.path.join(output_dir, 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    frames = []
    layer_names = sorted([k for k in layer_analysis['gradients'].keys() if 'blocks' in k],
                        key=lambda x: (x.split('.')[0], int(x.split('.')[1])))
    
    for idx, layer_name in enumerate(layer_names):
        # Skip some layers if too many
        if len(layer_names) > 20 and idx % 2 == 1:
            continue
            
        grad = layer_analysis['gradients'][layer_name]
        
        # Process gradient
        if grad.dim() == 3:
            grad_mag = grad.abs().mean(dim=-1)[0]
            if grad_mag.shape[0] > 196:
                patch_grad = grad_mag[1:]
            else:
                patch_grad = grad_mag
            
            # Ensure we're working with CPU tensors for numpy operations
            patch_grad = patch_grad.cpu()
            
            n_patches = int(np.sqrt(len(patch_grad)))
            if n_patches * n_patches == len(patch_grad):
                grad_2d = patch_grad.numpy().reshape(n_patches, n_patches)
            else:
                side = int(np.sqrt(len(patch_grad)))
                if side * side < len(patch_grad):
                    side += 1
                padded = torch.zeros(side * side)
                padded[:len(patch_grad)] = patch_grad
                grad_2d = padded.numpy().reshape(side, side)
            
            # Resize and normalize
            grad_resized = cv2.resize(grad_2d, (224, 224), interpolation=cv2.INTER_CUBIC)
            grad_resized = (grad_resized - grad_resized.min()) / (grad_resized.max() - grad_resized.min() + 1e-8)
            
            # Create frame
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            ax1.imshow(grad_resized, cmap='hot')
            ax1.set_title(f'Gradient Magnitude\nLayer: {layer_name}', fontsize=12)
            ax1.axis('off')
            
            ax2.imshow(image_np)
            ax2.imshow(grad_resized, cmap='hot', alpha=0.5)
            ax2.set_title('Overlay on Image', fontsize=12)
            ax2.axis('off')
            
            plt.suptitle(f'Layer {idx + 1}/{len(layer_names)}', fontsize=14)
            plt.tight_layout()
            
            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{idx:03d}.png')
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            frames.append(imageio.imread(frame_path))
    
    # Create GIF
    gif_path = os.path.join(output_dir, 'gradient_evolution.gif')
    imageio.mimsave(gif_path, frames, duration=0.5)
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Created gradient evolution GIF: {gif_path}")

def setup_device():
    """Set up computing device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def get_image_transform(input_size=224):
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def load_single_image(image_path, transform, device):
    """Load and preprocess a single image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)

def denormalize_image(image_tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    denormalized = image_tensor.cpu() * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized.permute(1, 2, 0).numpy()

def main():
    parser = argparse.ArgumentParser(description='CaiT gradient-based saliency analysis')
    parser.add_argument('--image_path', type=str, help='Path to a single image')
    parser.add_argument('--val_dir', type=str, help='ImageNet validation directory')
    parser.add_argument('--output_dir', type=str, default='./cait_saliency_analysis', 
                       help='Output directory')
    parser.add_argument('--model_name', type=str, default='cait_s24_224', 
                       help='CaiT model variant')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='Number of samples to process')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['vanilla', 'smooth', 'integrated'],
                       help='Saliency methods to use')
    
    args = parser.parse_args()
    
    if not args.image_path and not args.val_dir:
        raise ValueError("Please provide either --image_path or --val_dir")
    
    # Setup
    device = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading CaiT model: {args.model_name}")
    model = timm.create_model(args.model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Initialize gradient analyzer
    analyzer = GradientAnalyzer(model, device)
    
    # Get transform
    transform = get_image_transform()
    
    # Get class names (for ImageNet)
    try:
        with open('imagenet_classes.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Warning: imagenet_classes.txt not found. Class names will not be displayed.")
        class_names = []
    
    if args.image_path:
        # Process single image
        print(f"Processing image: {args.image_path}")
        image_tensor = load_single_image(args.image_path, transform, device)
        image_np = denormalize_image(image_tensor)
        
        # Compute saliency maps with different methods
        saliency_maps = []
        for method in args.methods:
            print(f"Computing {method} saliency...")
            saliency = analyzer.compute_saliency(image_tensor, method=method)
            saliency_maps.append(saliency)
        
        # Visualize saliency comparison
        comparison_path = os.path.join(args.output_dir, 'saliency_comparison.png')
        SaliencyVisualizer.visualize_saliency_comparison(
            image_np, saliency_maps, args.methods, comparison_path
        )
        
        # Analyze layer gradients
        print("Analyzing layer-wise gradients...")
        layer_analysis = analyzer.analyze_layer_gradients(image_tensor)
        
        # Get predicted class
        pred_class = layer_analysis['target_class']
        if isinstance(pred_class, torch.Tensor):
            pred_class = pred_class.item()
        if pred_class < len(class_names):
            print(f"Predicted class: {class_names[pred_class]}")
        
        # Visualize layer gradients
        grad_mags, act_mags = SaliencyVisualizer.visualize_layer_gradients(
            layer_analysis, image_np, args.output_dir
        )
        
        # Create gradient evolution GIF
        print("Creating gradient evolution GIF...")
        create_gradient_evolution_gif(analyzer, image_tensor, image_np, args.output_dir)
        
    else:
        # Process validation dataset
        val_dataset = ImageFolder(args.val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        for idx, (images, labels) in enumerate(val_loader):
            if idx >= args.num_samples:
                break
                
            print(f"\nProcessing sample {idx + 1}/{args.num_samples}")
            
            images = images.to(device)
            image_np = denormalize_image(images[0])
            
            # Create sample directory
            sample_dir = os.path.join(args.output_dir, f'sample_{idx}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Compute saliency maps
            saliency_maps = []
            for method in args.methods:
                print(f"  Computing {method} saliency...")
                # Ensure labels are on the same device as the model
                target = labels[0].to(device) if labels is not None else None
                saliency = analyzer.compute_saliency(images, target_class=target, method=method)
                saliency_maps.append(saliency)
            
            # Visualize
            comparison_path = os.path.join(sample_dir, 'saliency_comparison.png')
            SaliencyVisualizer.visualize_saliency_comparison(
                image_np, saliency_maps, args.methods, comparison_path
            )
            
            # Analyze layer gradients
            print("  Analyzing layer gradients...")
            target = labels[0].to(device) if labels is not None else None
            layer_analysis = analyzer.analyze_layer_gradients(images, target)
            
            # Visualize
            SaliencyVisualizer.visualize_layer_gradients(
                layer_analysis, image_np, sample_dir
            )
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    
    # Print summary of methods used
    print("\nSaliency Methods Used:")
    print("- Vanilla Gradient: Basic gradient with respect to input")
    print("- SmoothGrad: Averaged gradients with noise for smoother maps")
    print("- Integrated Gradients: Path-integrated gradients from baseline")

if __name__ == "__main__":
    main()