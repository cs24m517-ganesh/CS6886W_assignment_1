
import os
import random
import numpy as np
import torch
import torch.nn as nn

from utils.config import get_config
from utils.train_utils import prepare_dataloaders, select_optimizer, train_one_epoch, evaluate, save_checkpoint
from utils.wandb_logger import init_wandb, finish_wandb
from models.vgg6 import VGG6

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(arg_device):
    if arg_device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(arg_device)

def main():
    cfg = get_config()
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    wandb_obj = init_wandb(cfg)
    trainloader, testloader = prepare_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)
    model = VGG6(activation=cfg.activation, dropout=cfg.dropout, batch_norm=cfg.batch_norm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = select_optimizer(cfg.optimizer, model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    best_val, best_path = 0, None
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        print(f'Epoch {epoch}/{cfg.epochs} - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

        if wandb_obj:
            wandb_obj.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss,
                           'val_acc': val_acc, 'val_loss': val_loss})

        if val_acc > best_val:
            best_val = val_acc
            best_path = save_checkpoint({'epoch': epoch,
                                         'model_state': model.state_dict(),
                                         'optimizer_state': optimizer.state_dict(),
                                         'val_acc': val_acc,
                                         'cfg': vars(cfg)},
                                        cfg.save_dir, f'best_val_{val_acc:.2f}.pth')
            print(f'✅ New best model: {best_val:.2f}% saved at {best_path}')

    print(f'Best model saved to {best_path}')
    finish_wandb(wandb_obj)

if __name__ == '__main__':
    main()
