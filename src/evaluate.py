"""
Evaluation script for MNIST model
This script demonstrates model evaluation and metrics reporting
"""
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Load a trained model from checkpoint"""
    model = get_model(
        num_classes=config['model']['num_classes'],
        device=device
    )
    
    # Try loading from checkpoint first
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint: {model_path}")
            if 'val_accuracy' in checkpoint:
                print(f"Checkpoint validation accuracy: {checkpoint['val_accuracy']:.2f}%")
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model weights from: {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    return model


def get_test_loader(config):
    """Create test data loader"""
    # Resolve data directory path
    data_dir = config['data']['data_dir']
    if not os.path.isabs(data_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, data_dir)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    return test_loader


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy, np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, save_path='logs/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
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
    
    # Determine which model to load
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'latest.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config['paths']['model_dir'], 'final_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print("No trained model found. Please run train.py first.")
        return
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, config, device)
    
    # Get test loader
    print("Loading test data...")
    test_loader = get_test_loader(config)
    
    # Evaluate
    print("Evaluating model...")
    accuracy, predictions, targets = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=[str(i) for i in range(10)]))
    
    # Plot confusion matrix
    plot_confusion_matrix(targets, predictions)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
