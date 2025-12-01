# Text-to-Image Synthesis with GANs

A reproducible implementation of text-to-image synthesis using Generative Adversarial Networks (GANs) with BERT text embeddings. This project generates images from text descriptions using a conditional GAN architecture trained on the CIFAR-10 dataset.

## Features

- **Modern Architecture**: Spectral normalization, proper conditioning, and stable training
- **BERT Integration**: Uses pre-trained BERT for high-quality text embeddings
- **Reproducible**: Deterministic seeding and comprehensive configuration management
- **Multi-Device Support**: Automatic device detection (CUDA/MPS/CPU)
- **Interactive Demo**: Streamlit web application for easy experimentation
- **Comprehensive Evaluation**: FID, IS, and other quality metrics
- **Production Ready**: Clean code structure, type hints, and documentation

## Project Structure

```
├── src/
│   ├── models/           # Model implementations
│   ├── data/            # Data loading and preprocessing
│   ├── configs/         # Configuration files
│   └── utils/           # Utility functions
├── scripts/             # Training and sampling scripts
├── demo/               # Streamlit demo application
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks for experimentation
├── assets/             # Generated samples and visualizations
└── requirements.txt    # Python dependencies
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Text-to-Image-Synthesis-with-GANs.git
cd Text-to-Image-Synthesis-with-GANs

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Train the model with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config-name custom_config
```

### 3. Sampling

```bash
# Generate samples from trained model
python scripts/sample.py \
    --checkpoint checkpoints/best_model.ckpt \
    --texts "a photo of a cat" "a photo of a dog" \
    --num_samples 4 \
    --output_dir outputs
```

### 4. Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `src/configs/config.yaml`: Main configuration
- `src/configs/model/text_to_image_gan.yaml`: Model architecture
- `src/configs/data/cifar10_captions.yaml`: Data loading settings
- `src/configs/training/default.yaml`: Training parameters

### Example Configuration Override

```bash
python scripts/train.py \
    model.z_dim=128 \
    training.max_epochs=200 \
    training.optimizer.generator.lr=0.0001
```

## Model Architecture

### Generator
- Input: Noise vector (z) + BERT text embeddings
- Architecture: Fully connected layers with spectral normalization
- Output: 64x64 RGB images

### Discriminator
- Input: Images + BERT text embeddings
- Architecture: Fully connected layers with spectral normalization
- Output: Real/fake classification

### Text Encoder
- Pre-trained BERT model (frozen weights)
- Text pooling: Mean of token embeddings
- Output: 768-dimensional text embeddings

## Dataset

The model is trained on CIFAR-10 with automatically generated text captions:

- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Image Size**: 32x32 → resized to 64x64
- **Captions**: Multiple templates per class for diversity
- **Augmentation**: Random horizontal flip, color jitter

## Training

### Key Features
- **Spectral Normalization**: Stabilizes GAN training
- **Learning Rate Scheduling**: Cosine annealing
- **Mixed Precision**: Automatic mixed precision for efficiency
- **Gradient Clipping**: Prevents gradient explosion
- **Deterministic Training**: Reproducible results

### Training Command
```bash
python scripts/train.py \
    training.max_epochs=100 \
    training.optimizer.generator.lr=0.0002 \
    training.optimizer.discriminator.lr=0.0002 \
    logging.use_wandb=true
```

## Evaluation

### Metrics
- **FID (Fréchet Inception Distance)**: Quality and diversity
- **IS (Inception Score)**: Quality assessment
- **Precision/Recall**: Generative model evaluation

### Evaluation Command
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.ckpt \
    --data_dir data/ \
    --output_dir evaluation_results/
```

## Sampling and Generation

### Command Line Interface
```bash
python scripts/sample.py \
    --checkpoint checkpoints/best_model.ckpt \
    --texts "a photo of a cat" "a cute dog" \
    --num_samples 8 \
    --seed 42 \
    --output_dir samples/
```

### Programmatic Usage
```python
from src.models.text_to_image_gan import TextToImageGAN
import torch

# Load model
model = TextToImageGAN()
model.load_state_dict(torch.load("checkpoint.ckpt")["state_dict"])

# Generate images
z = torch.randn(4, 100)
texts = ["a photo of a cat"] * 4
images = model.generate(z, texts)
```

## Interactive Demo

The Streamlit demo provides an intuitive interface for:

- **Text Input**: Enter custom prompts
- **Parameter Control**: Adjust seed, device, etc.
- **Real-time Generation**: Instant image generation
- **Download**: Save generated images
- **Model Info**: View architecture details

Launch with:
```bash
streamlit run demo/app.py
```

## Device Support

### Automatic Device Detection
- **CUDA**: NVIDIA GPUs with CUDA support
- **MPS**: Apple Silicon GPUs (M1/M2)
- **CPU**: Fallback for all systems

### Manual Device Selection
```python
from src.utils.device import get_device

device = get_device("cuda")  # Force CUDA
device = get_device("mps")    # Force MPS
device = get_device("cpu")    # Force CPU
```

## Reproducibility

### Seeding
```python
from src.utils.device import set_seed

set_seed(42)  # Ensures reproducible results
```

### Deterministic Training
- Fixed random seeds for all libraries
- Deterministic CUDA operations
- Reproducible data loading

## Development

### Code Quality
- **Type Hints**: Full type annotation
- **Documentation**: Google-style docstrings
- **Formatting**: Black + Ruff
- **Testing**: Pytest with coverage

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v --cov=src/
```

## Performance

### Training Time
- **CUDA**: ~2-3 hours for 100 epochs
- **MPS**: ~3-4 hours for 100 epochs  
- **CPU**: ~8-10 hours for 100 epochs

### Memory Usage
- **Training**: ~4-6 GB GPU memory
- **Inference**: ~2-3 GB GPU memory
- **CPU**: ~8-12 GB RAM

## Limitations

- **Image Resolution**: Limited to 64x64 pixels
- **Dataset**: Trained only on CIFAR-10 classes
- **Text Complexity**: Simple, template-based captions
- **Quality**: Basic GAN architecture (not diffusion-based)

## Future Improvements

- **Higher Resolution**: Progressive GAN or StyleGAN
- **Better Text**: CLIP embeddings or T5
- **More Data**: COCO or Conceptual Captions
- **Advanced Architecture**: Diffusion models or VQGAN

## Citation

If you use this code in your research, please cite:

```bibtex
@software{text_to_image_gan,
  title={Text-to-Image Synthesis with GANs},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Text-to-Image-Synthesis-with-GANs}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- CIFAR-10 dataset creators
- Hugging Face Transformers team
- PyTorch Lightning developers
- Streamlit team
# Text-to-Image-Synthesis-with-GANs
