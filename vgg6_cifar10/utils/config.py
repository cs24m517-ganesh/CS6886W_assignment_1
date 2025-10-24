
import argparse

def get_config():
    parser = argparse.ArgumentParser(description='Train VGG6 on CIFAR-10')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_norm', type=lambda x: str(x).lower() == 'true', default=False)

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='vgg6_cifar10')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')

    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=1)

    return parser.parse_args()
