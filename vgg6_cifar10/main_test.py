import os
import torch
import argparse
from utils.config import get_config
from utils.train_utils import prepare_dataloaders
from utils.test_utils import test_model
from models.vgg6 import VGG6


def parse_args():
    """Parse arguments including checkpoint and save directory."""
    parser = argparse.ArgumentParser(description="Test VGG6 model on CIFAR-10")
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (.pth)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory containing checkpoints')
    parser.add_argument('--device', type=str, default='auto', help='cpu, cuda, or auto')
    return parser.parse_args()


def main():
    # Load arguments
    cfg = parse_args()

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.device == 'auto' else cfg.device)
    print(f" Using device: {device}")

    # Prepare CIFAR-10 test data
    _, testloader = prepare_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)

    # Find checkpoint
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        ckpt_path = cfg.checkpoint
    else:
        files = sorted([f for f in os.listdir(cfg.save_dir) if f.endswith('.pth')])
        if not files:
            print(f" No checkpoint found in directory: {cfg.save_dir}")
            return
        ckpt_path = os.path.join(cfg.save_dir, files[-1])

    print(f" Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load model configuration
    activation = ckpt['cfg'].get('activation', 'gelu')
    batch_norm = ckpt['cfg'].get('batch_norm', True)

    # Initialize model
    model = VGG6(activation=activation, batch_norm=batch_norm).to(device)
    model.load_state_dict(ckpt['model_state'])

    # Evaluate model using your test utility
    acc, cm = test_model(model, testloader, device)

    # Display results
    print(f"\n Test Accuracy: {acc:.2f}%")
    print("\n Confusion Matrix:")
    print(cm)


if __name__ == '__main__':
    main()
