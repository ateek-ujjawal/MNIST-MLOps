"""
Training script for MNIST model
This script demonstrates MLOps best practices:
- Configuration management
- Logging and monitoring
- Model checkpointing
- Reproducibility
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from datetime import datetime

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


def get_data_loaders(config):
    """
    Create data loaders for training and validation.
    Demonstrates data pipeline management.
    """
    # Resolve data directory path
    data_dir = config['data']['data_dir']
    if not os.path.isabs(data_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, data_dir)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load dataset
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=config['data']['download'],
        transform=transform
    )
    
    # Split into train and validation
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
        
        # Log to tensorboard
        if batch_idx % config['logging']['log_interval'] == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy', 100.*correct/total, global_step)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch, writer):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/Accuracy', val_acc, epoch)
    
    return val_loss, val_acc


def save_checkpoint(model, optimizer, epoch, val_acc, config, is_best=False):
    """Save model checkpoint"""
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'config': config
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(config['paths']['checkpoint_dir'], 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model with validation accuracy: {val_acc:.2f}%")


def main():
    # Load configuration
    config = load_config()
    
    # Get project root for resolving relative paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve relative paths in config
    for key in ['data_dir', 'model_dir', 'log_dir', 'checkpoint_dir']:
        if key in config.get('paths', {}):
            path = config['paths'][key]
            if not os.path.isabs(path):
                config['paths'][key] = os.path.join(project_root, path)
        elif key == 'data_dir' and key in config.get('data', {}):
            path = config['data'][key]
            if not os.path.isabs(path):
                config['data'][key] = os.path.join(project_root, path)
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config['paths']['log_dir'], f"run_{timestamp}")
    writer = SummaryWriter(log_dir) if config['logging']['use_tensorboard'] else None
    
    # Save config to log directory
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(config)
    
    # Initialize model
    print("Initializing model...")
    model = get_model(
        num_classes=config['model']['num_classes'],
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Training loop
    best_val_acc = 0.0
    print("\nStarting training...")
    print(f"Training for {config['training']['num_epochs']} epochs")
    print("-" * 50)
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, config
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        if (epoch + 1) % config['logging']['save_checkpoint_interval'] == 0:
            save_checkpoint(model, optimizer, epoch, val_acc, config, is_best)
        
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)
    
    # Save final model
    final_model_path = os.path.join(config['paths']['model_dir'], 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
