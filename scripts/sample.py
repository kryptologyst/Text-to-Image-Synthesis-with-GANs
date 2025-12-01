#!/usr/bin/env python3
"""Sampling script for text-to-image GAN."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt

from src.models.text_to_image_gan import TextToImageGAN
from src.utils.device import get_device, set_seed


def load_model(checkpoint_path: str, device: torch.device) -> TextToImageGAN:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint
    model_config = checkpoint["hyper_parameters"]["model_config"]
    
    # Initialize model
    model = TextToImageGAN(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    return model


def generate_samples(
    model: TextToImageGAN,
    texts: List[str],
    num_samples: int = 1,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate samples from model.
    
    Args:
        model: Trained model
        texts: List of text prompts
        num_samples: Number of samples per text
        seed: Random seed
        device: Device to use
        
    Returns:
        Generated images tensor
    """
    if seed is not None:
        set_seed(seed)
    
    model.eval()
    with torch.no_grad():
        # Generate noise
        z = torch.randn(len(texts) * num_samples, model.generator.z_dim, device=device)
        
        # Repeat texts for multiple samples
        repeated_texts = [text for text in texts for _ in range(num_samples)]
        
        # Generate images
        generated_images = model.generate(z, repeated_texts)
        
    return generated_images


def save_samples(
    images: torch.Tensor,
    texts: List[str],
    output_dir: str,
    filename: str = "samples.png",
    grid_size: int = 4
) -> None:
    """Save generated samples as image grid.
    
    Args:
        images: Generated images tensor
        texts: Text prompts
        output_dir: Output directory
        filename: Output filename
        grid_size: Size of grid (nrow)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create grid
    grid = vutils.make_grid(
        images,
        nrow=grid_size,
        normalize=True,
        value_range=(-1, 1)
    )
    
    # Convert to PIL Image
    grid_np = grid.cpu().permute(1, 2, 0).numpy()
    grid_np = (grid_np + 1) / 2  # Denormalize
    grid_np = (grid_np * 255).astype("uint8")
    
    grid_image = Image.fromarray(grid_np)
    
    # Save image
    output_path = os.path.join(output_dir, filename)
    grid_image.save(output_path)
    print(f"Saved samples to {output_path}")


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Generate samples from text-to-image GAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--texts", nargs="+", required=True, help="Text prompts for generation")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per text")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--filename", type=str, default="samples.png", help="Output filename")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--grid_size", type=int, default=4, help="Grid size for display")
    
    args = parser.parse_args()
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Generate samples
    print(f"Generating samples for texts: {args.texts}")
    generated_images = generate_samples(
        model=model,
        texts=args.texts,
        num_samples=args.num_samples,
        seed=args.seed,
        device=device
    )
    
    # Save samples
    save_samples(
        images=generated_images,
        texts=args.texts,
        output_dir=args.output_dir,
        filename=args.filename,
        grid_size=args.grid_size
    )
    
    print("Sampling completed!")


if __name__ == "__main__":
    main()
