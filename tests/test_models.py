"""Unit tests for text-to-image GAN components."""

import pytest
import torch
import torch.nn as nn

from src.models.text_to_image_gan import (
    SpectralNorm,
    TextEncoder,
    Generator,
    Discriminator,
    TextToImageGAN
)
from src.utils.device import get_device, set_seed


class TestSpectralNorm:
    """Test spectral normalization layer."""
    
    def test_spectral_norm_initialization(self):
        """Test spectral norm initialization."""
        linear = nn.Linear(10, 5)
        sn_linear = SpectralNorm(linear)
        
        assert hasattr(sn_linear, 'u')
        assert hasattr(sn_linear, 'v')
        assert sn_linear.u.size(0) == 5
        assert sn_linear.v.size(0) == 10
    
    def test_spectral_norm_forward(self):
        """Test spectral norm forward pass."""
        linear = nn.Linear(10, 5)
        sn_linear = SpectralNorm(linear)
        
        x = torch.randn(3, 10)
        output = sn_linear(x)
        
        assert output.shape == (3, 5)


class TestTextEncoder:
    """Test text encoder."""
    
    def test_text_encoder_initialization(self):
        """Test text encoder initialization."""
        encoder = TextEncoder()
        
        assert encoder.model is not None
        assert encoder.tokenizer is not None
        assert encoder.pooling_strategy == "mean"
    
    def test_text_encoding(self):
        """Test text encoding."""
        encoder = TextEncoder()
        texts = ["a photo of a cat", "a photo of a dog"]
        
        embeddings = encoder.encode_text(texts)
        
        assert embeddings.shape == (2, 768)
        assert embeddings.dtype == torch.float32


class TestGenerator:
    """Test generator network."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = Generator()
        
        assert generator.z_dim == 100
        assert generator.text_embedding_dim == 768
        assert generator.img_channels == 3
        assert generator.img_size == 64
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = Generator()
        
        batch_size = 4
        z = torch.randn(batch_size, generator.z_dim)
        text_embeddings = torch.randn(batch_size, generator.text_embedding_dim)
        
        output = generator(z, text_embeddings)
        
        assert output.shape == (batch_size, 3, 64, 64)
        assert output.min() >= -1.0
        assert output.max() <= 1.0


class TestDiscriminator:
    """Test discriminator network."""
    
    def test_discriminator_initialization(self):
        """Test discriminator initialization."""
        discriminator = Discriminator()
        
        assert discriminator.text_embedding_dim == 768
        assert discriminator.img_channels == 3
        assert discriminator.img_size == 64
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = Discriminator()
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 64, 64)
        text_embeddings = torch.randn(batch_size, discriminator.text_embedding_dim)
        
        output = discriminator(images, text_embeddings)
        
        assert output.shape == (batch_size, 1)
        assert output.min() >= 0.0
        assert output.max() <= 1.0


class TestTextToImageGAN:
    """Test complete text-to-image GAN."""
    
    def test_gan_initialization(self):
        """Test GAN initialization."""
        gan = TextToImageGAN()
        
        assert gan.text_encoder is not None
        assert gan.generator is not None
        assert gan.discriminator is not None
    
    def test_text_encoding(self):
        """Test text encoding through GAN."""
        gan = TextToImageGAN()
        texts = ["a photo of a cat"]
        
        embeddings = gan.encode_text(texts)
        
        assert embeddings.shape == (1, 768)
    
    def test_generation(self):
        """Test image generation."""
        gan = TextToImageGAN()
        
        batch_size = 2
        z = torch.randn(batch_size, gan.generator.z_dim)
        texts = ["a photo of a cat", "a photo of a dog"]
        
        images = gan.generate(z, texts)
        
        assert images.shape == (batch_size, 3, 64, 64)
    
    def test_discrimination(self):
        """Test image discrimination."""
        gan = TextToImageGAN()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 64, 64)
        texts = ["a photo of a cat", "a photo of a dog"]
        
        scores = gan.discriminate(images, texts)
        
        assert scores.shape == (batch_size, 1)


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(5)
        numpy_rand = torch.rand(5).numpy()
        
        # Set seed again and generate
        set_seed(42)
        torch_rand2 = torch.rand(5)
        numpy_rand2 = torch.rand(5).numpy()
        
        # Should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert torch.allclose(torch.tensor(numpy_rand), torch.tensor(numpy_rand2))


if __name__ == "__main__":
    pytest.main([__file__])
