"""
Inference script for making predictions on new images
This demonstrates model serving and inference in production
"""
import os
import sys
import yaml
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import get_model


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file"""
    # Handle both relative and absolute paths
    if not os.path.isabs(config_path):
        # Try relative to project root (parent of src)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path, config, device):
    """Load a trained model"""
    model = get_model(
        num_classes=config['model']['num_classes'],
        device=device
    )
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    return model


def preprocess_image(image_path):
    """
    Preprocess image for inference.
    This is a critical MLOps practice: ensuring consistent preprocessing.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale if needed
        transforms.Resize((28, 28)),  # Resize to MNIST size
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Same normalization as training
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def predict(model, image_tensor, device):
    """Make prediction on a single image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()


def visualize_prediction(image_path, predicted_class, confidence, probabilities):
    """Visualize the prediction"""
    image = Image.open(image_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show image
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Predicted: {predicted_class} (Confidence: {confidence:.2%})')
    ax1.axis('off')
    
    # Show probabilities
    ax2.bar(range(10), probabilities)
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.set_xticks(range(10))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/prediction_visualization.png')
    print("Visualization saved to logs/prediction_visualization.png")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MNIST Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default=None, help='Path to model file (optional)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Get project root for resolving relative paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve relative paths in config
    for key in ['model_dir', 'checkpoint_dir']:
        if key in config.get('paths', {}):
            path = config['paths'][key]
            if not os.path.isabs(path):
                config['paths'][key] = os.path.join(project_root, path)
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        # Try to find best model
        checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best.pth')
        if os.path.exists(checkpoint_path):
            model_path = checkpoint_path
        else:
            model_path = os.path.join(config['paths']['model_dir'], 'final_model.pth')
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, config, device)
    
    # Preprocess image
    print(f"Loading and preprocessing image: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # Make prediction
    predicted_class, confidence, probabilities = predict(model, image_tensor, device)
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted Digit: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {i}: {prob:.4f}")
    
    # Visualize
    os.makedirs('logs', exist_ok=True)
    visualize_prediction(args.image, predicted_class, confidence, probabilities)
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()
