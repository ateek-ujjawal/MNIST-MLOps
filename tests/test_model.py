"""Tests for model module."""
import sys
import os

import torch

# Add project root so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model, SimpleCNN


def test_model_creation():
    """Model can be created with default args."""
    model = get_model(num_classes=10, device="cpu")
    assert model is not None
    assert isinstance(model, SimpleCNN)


def test_forward_pass():
    """Model forward pass works with correct input shape (batch, 1, 28, 28)."""
    model = get_model(num_classes=10, device="cpu")
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)
