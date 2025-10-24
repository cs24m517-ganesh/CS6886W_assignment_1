import os
import torch
import argparse
from utils.train_utils import prepare_dataloaders
from models.vgg6 import VGG6


def parse_args():
    """Parse arguments including checkpoint and basic config."""
    parser = argparse.ArgumentParser(description="Evaluate VGG6 on CIFAR-10")
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (.pth)')
    parser.add_argument('--device', type=str, default='auto', help='cpu, cuda, or auto')
    return parser.parse_args()


def main():
    # Parse arguments
    cfg = parse_args()

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.device == 'auto' else cfg.device)
    print(f" Using device: {device}")

    # Prepare dataloaders
    _, testloader = prepare_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)

    # Resolve checkpoint path
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        ckpt_path = cfg.checkpoint
    else:
        ckpt_dir = './checkpoints'
        files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        if not files:
            print(" No checkpoint found.")
            return
        ckpt_path = os.path.join(ckpt_dir, sorted(files)[-1])

    print(f" Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load model
    activation = ckpt.get('cfg', {}).get('activation', 'gelu')
    batch_norm = ckpt.get('cfg', {}).get('batch_norm', True)
    model = VGG6(activation=activation, batch_norm=batch_norm).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Evaluate
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f" Validation Accuracy: {acc:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
