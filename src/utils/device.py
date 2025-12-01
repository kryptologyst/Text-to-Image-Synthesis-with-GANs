"""Utility functions for device management and reproducibility."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu")
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device_count": 0,
        "current_device": None,
        "device_name": None
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device_name"] = "Apple Silicon GPU (MPS)"
    
    return info


def print_device_info() -> None:
    """Print device information."""
    info = get_device_info()
    print("Device Information:")
    print(f"  CUDA available: {info['cuda_available']}")
    print(f"  MPS available: {info['mps_available']}")
    if info["device_name"]:
        print(f"  Device: {info['device_name']}")
    if info["device_count"] > 0:
        print(f"  Device count: {info['device_count']}")
        print(f"  Current device: {info['current_device']}")
