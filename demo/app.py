"""Streamlit demo application for text-to-image GAN."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import streamlit as st
import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np

from src.models.text_to_image_gan import TextToImageGAN
from src.utils.device import get_device, set_seed


@st.cache_resource
def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint with caching."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["hyper_parameters"]["model_config"]
    
    model = TextToImageGAN(**model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    return model


def generate_image(model, text: str, seed: int, device: torch.device) -> Image.Image:
    """Generate image from text prompt."""
    set_seed(seed)
    
    with torch.no_grad():
        z = torch.randn(1, model.generator.z_dim, device=device)
        generated_image = model.generate(z, [text])
        
        # Convert to PIL Image
        img = generated_image[0].cpu()
        img = (img + 1) / 2  # Denormalize
        img = torch.clamp(img, 0, 1)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        return Image.fromarray(img_np)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Text-to-Image GAN Demo",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Text-to-Image GAN Demo")
    st.markdown("Generate images from text descriptions using a trained GAN model.")
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Model selection
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if checkpoint_files:
            selected_checkpoint = st.sidebar.selectbox(
                "Select Model Checkpoint",
                checkpoint_files,
                index=0
            )
            checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
        else:
            st.error("No checkpoint files found in checkpoints directory")
            return
    else:
        st.error("Checkpoints directory not found")
        return
    
    # Device selection
    device_options = ["auto", "cpu"]
    if torch.cuda.is_available():
        device_options.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_options.append("mps")
    
    selected_device = st.sidebar.selectbox("Device", device_options, index=0)
    device = get_device(selected_device)
    
    # Text input
    text_prompt = st.sidebar.text_area(
        "Text Prompt",
        value="a photo of a cat",
        height=100,
        help="Enter a text description for the image you want to generate"
    )
    
    # Seed control
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=2**32-1,
        value=42,
        help="Set a random seed for reproducible generation"
    )
    
    # Generate button
    generate_button = st.sidebar.button("Generate Image", type="primary")
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model(checkpoint_path, device)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Generated Image")
        
        if generate_button and text_prompt.strip():
            with st.spinner("Generating image..."):
                try:
                    generated_image = generate_image(model, text_prompt, seed, device)
                    st.image(generated_image, caption=f"Generated: {text_prompt}", use_column_width=True)
                    
                    # Download button
                    img_bytes = generated_image.tobytes()
                    st.download_button(
                        label="Download Image",
                        data=img_bytes,
                        file_name=f"generated_{text_prompt.replace(' ', '_')}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")
        else:
            st.info("Click 'Generate Image' to create an image from your text prompt.")
    
    with col2:
        st.header("Model Information")
        
        # Display model info
        st.subheader("Model Configuration")
        model_config = {
            "Z Dimension": model.generator.z_dim,
            "Text Embedding Dimension": model.text_encoder.model.config.hidden_size,
            "Image Size": f"{model.generator.img_size}x{model.generator.img_size}",
            "Image Channels": model.generator.img_channels,
        }
        
        for key, value in model_config.items():
            st.write(f"**{key}**: {value}")
        
        # Device info
        st.subheader("Device Information")
        st.write(f"**Current Device**: {device}")
        st.write(f"**CUDA Available**: {torch.cuda.is_available()}")
        if hasattr(torch.backends, "mps"):
            st.write(f"**MPS Available**: {torch.backends.mps.is_available()}")
    
    # Example prompts
    st.header("Example Prompts")
    example_prompts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a car",
        "a photo of an airplane",
        "a photo of a ship",
        "a photo of a horse",
        "a photo of a truck",
        "a cute cat",
        "a friendly dog",
        "a colorful bird",
        "a red car",
        "a white airplane",
        "a blue ship"
    ]
    
    cols = st.columns(3)
    for i, prompt in enumerate(example_prompts):
        with cols[i % 3]:
            if st.button(prompt, key=f"prompt_{i}"):
                st.session_state.text_prompt = prompt
                st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note**: This demo uses a text-to-image GAN trained on CIFAR-10 dataset. "
        "The model generates 64x64 images based on text descriptions."
    )


if __name__ == "__main__":
    main()
