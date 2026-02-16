"""
Simple CNN model for MNIST classification
This is a basic convolutional neural network architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
    - Fully connected layers: 64*7*7 -> 128 -> 10
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After two pooling operations: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(num_classes=10, device='cpu'):
    """
    Factory function to create and return a model instance.
    
    Args:
        num_classes: Number of output classes (10 for MNIST)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Model instance
    """
    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)
    return model
