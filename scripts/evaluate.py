#!/usr/bin/env python3
"""Evaluation script for text-to-image GAN."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np
from clean_fid import fid
from torchmetrics.image import FrechetInceptionDistance, InceptionScore

from src.models.text_to_image_gan import TextToImageGAN
from src.data.cifar10_captions import CIFAR10CaptionsDataset
from src.utils.device import get_device, set_seed


def load_model(checkpoint_path: str, device: torch.device) -> TextToImageGAN:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["hyper_parameters"]["model_config"]
    
    model = TextToImageGAN(**model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    return model


def generate_samples_for_evaluation(
    model: TextToImageGAN,
    num_samples: int,
    device: torch.device,
    seed: int = 42
) -> torch.Tensor:
    """Generate samples for evaluation."""
    set_seed(seed)
    
    # Use diverse captions for evaluation
    captions = [
        "a photo of a cat", "a photo of a dog", "a photo of a bird",
        "a photo of a car", "a photo of an airplane", "a photo of a ship",
        "a photo of a horse", "a photo of a truck", "a photo of a deer",
        "a photo of a frog"
    ]
    
    all_samples = []
    samples_per_caption = num_samples // len(captions)
    
    model.eval()
    with torch.no_grad():
        for caption in captions:
            for _ in range(samples_per_caption):
                z = torch.randn(1, model.generator.z_dim, device=device)
                sample = model.generate(z, [caption])
                all_samples.append(sample)
    
    return torch.cat(all_samples, dim=0)


def calculate_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """Calculate FID score."""
    # Convert to numpy arrays
    real_np = real_images.cpu().numpy()
    fake_np = fake_images.cpu().numpy()
    
    # Normalize to [0, 255]
    real_np = ((real_np + 1) / 2 * 255).astype(np.uint8)
    fake_np = ((fake_np + 1) / 2 * 255).astype(np.uint8)
    
    # Calculate FID
    fid_score = fid.compute_fid(
        real_np, fake_np, 
        batch_size=64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    return fid_score


def calculate_inception_score(fake_images: torch.Tensor) -> float:
    """Calculate Inception Score."""
    # Normalize to [0, 1]
    fake_images = (fake_images + 1) / 2
    fake_images = torch.clamp(fake_images, 0, 1)
    
    # Calculate IS
    is_metric = InceptionScore(normalize=True)
    is_score = is_metric(fake_images)
    
    return is_score.item()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate text-to-image GAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Load real images
    print("Loading real images...")
    dataset = CIFAR10CaptionsDataset(
        root=args.data_dir,
        train=False,
        download=False,
        transform=None
    )
    
    # Get real images
    real_images = []
    for i in range(min(args.num_samples, len(dataset))):
        image, _ = dataset[i]
        real_images.append(image)
    
    real_images = torch.stack(real_images)
    real_images = real_images.to(device)
    
    # Generate fake images
    print(f"Generating {args.num_samples} samples...")
    fake_images = generate_samples_for_evaluation(
        model, args.num_samples, device, args.seed
    )
    
    # Calculate metrics
    print("Calculating FID...")
    fid_score = calculate_fid(real_images, fake_images)
    
    print("Calculating Inception Score...")
    is_score = calculate_inception_score(fake_images)
    
    # Save results
    results = {
        "fid": fid_score,
        "inception_score": is_score,
        "num_samples": args.num_samples,
        "checkpoint": args.checkpoint
    }
    
    results_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=================\n")
        f.write(f"FID Score: {fid_score:.4f}\n")
        f.write(f"Inception Score: {is_score:.4f}\n")
        f.write(f"Number of Samples: {args.num_samples}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
    
    print(f"\nEvaluation Results:")
    print(f"FID Score: {fid_score:.4f}")
    print(f"Inception Score: {is_score:.4f}")
    print(f"Results saved to {results_file}")
    
    # Save sample images
    sample_images = fake_images[:16]  # First 16 samples
    grid = vutils.make_grid(
        sample_images,
        nrow=4,
        normalize=True,
        value_range=(-1, 1)
    )
    
    sample_file = os.path.join(args.output_dir, "sample_images.png")
    vutils.save_image(grid, sample_file)
    print(f"Sample images saved to {sample_file}")


if __name__ == "__main__":
    main()
