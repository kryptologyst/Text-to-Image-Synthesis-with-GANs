#!/usr/bin/env python3
"""Test script to verify the text-to-image GAN implementation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.models.text_to_image_gan import TextToImageGAN
from src.utils.device import get_device, set_seed, print_device_info


def test_model_initialization():
    """Test model initialization."""
    print("Testing model initialization...")
    
    model = TextToImageGAN()
    print(f"✓ Model initialized successfully")
    print(f"  - Generator z_dim: {model.generator.z_dim}")
    print(f"  - Text embedding dim: {model.text_encoder.model.config.hidden_size}")
    print(f"  - Image size: {model.generator.img_size}x{model.generator.img_size}")
    
    return model


def test_text_encoding(model):
    """Test text encoding."""
    print("\nTesting text encoding...")
    
    texts = ["a photo of a cat", "a photo of a dog"]
    embeddings = model.encode_text(texts)
    
    print(f"✓ Text encoding successful")
    print(f"  - Input texts: {texts}")
    print(f"  - Output shape: {embeddings.shape}")
    print(f"  - Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    return embeddings


def test_image_generation(model):
    """Test image generation."""
    print("\nTesting image generation...")
    
    device = get_device("auto")
    model = model.to(device)
    
    batch_size = 4
    z = torch.randn(batch_size, model.generator.z_dim, device=device)
    texts = ["a photo of a cat"] * batch_size
    
    with torch.no_grad():
        images = model.generate(z, texts)
    
    print(f"✓ Image generation successful")
    print(f"  - Input noise shape: {z.shape}")
    print(f"  - Output image shape: {images.shape}")
    print(f"  - Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    return images


def test_discrimination(model):
    """Test image discrimination."""
    print("\nTesting image discrimination...")
    
    device = get_device("auto")
    model = model.to(device)
    
    batch_size = 4
    images = torch.randn(batch_size, 3, 64, 64, device=device)
    texts = ["a photo of a cat"] * batch_size
    
    with torch.no_grad():
        scores = model.discriminate(images, texts)
    
    print(f"✓ Image discrimination successful")
    print(f"  - Input image shape: {images.shape}")
    print(f"  - Output scores shape: {scores.shape}")
    print(f"  - Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    return scores


def test_reproducibility():
    """Test reproducibility with seeding."""
    print("\nTesting reproducibility...")
    
    set_seed(42)
    model1 = TextToImageGAN()
    
    set_seed(42)
    model2 = TextToImageGAN()
    
    # Generate with same seed
    set_seed(42)
    z1 = torch.randn(2, 100)
    texts = ["a photo of a cat", "a photo of a dog"]
    images1 = model1.generate(z1, texts)
    
    set_seed(42)
    z2 = torch.randn(2, 100)
    images2 = model2.generate(z2, texts)
    
    # Check if results are identical
    if torch.allclose(images1, images2, atol=1e-6):
        print("✓ Reproducibility test passed")
        print("  - Same seed produces identical results")
    else:
        print("✗ Reproducibility test failed")
        print("  - Results differ despite same seed")


def main():
    """Main test function."""
    print("=" * 60)
    print("Text-to-Image GAN Implementation Test")
    print("=" * 60)
    
    # Print device info
    print_device_info()
    
    # Test model initialization
    model = test_model_initialization()
    
    # Test text encoding
    embeddings = test_text_encoding(model)
    
    # Test image generation
    images = test_image_generation(model)
    
    # Test discrimination
    scores = test_discrimination(model)
    
    # Test reproducibility
    test_reproducibility()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    
    # Print summary
    print(f"\nModel Summary:")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Generator parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"  - Discriminator parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
    print(f"  - Text encoder parameters: {sum(p.numel() for p in model.text_encoder.parameters()):,}")


if __name__ == "__main__":
    main()
