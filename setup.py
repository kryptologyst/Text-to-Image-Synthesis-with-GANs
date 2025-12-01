#!/usr/bin/env python3
"""Setup script for text-to-image GAN project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "outputs", 
        "checkpoints",
        "logs",
        "assets",
        "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def setup_pre_commit():
    """Setup pre-commit hooks."""
    if run_command("pip install pre-commit", "Installing pre-commit"):
        return run_command("pre-commit install", "Installing pre-commit hooks")
    return False


def test_installation():
    """Test the installation."""
    return run_command("python scripts/test_implementation.py", "Testing installation")


def main():
    """Main setup function."""
    print("🚀 Setting up Text-to-Image GAN project...")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please check requirements.txt")
        return False
    
    # Setup pre-commit (optional)
    print("\n🔧 Setting up pre-commit hooks...")
    setup_pre_commit()
    
    # Test installation
    print("\n🧪 Testing installation...")
    if not test_installation():
        print("❌ Installation test failed. Please check the error messages above.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    
    print("\n📋 Next steps:")
    print("1. Train the model: python scripts/train.py")
    print("2. Generate samples: python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --texts 'a photo of a cat'")
    print("3. Launch demo: streamlit run demo/app.py")
    print("4. Run tests: pytest tests/ -v")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
