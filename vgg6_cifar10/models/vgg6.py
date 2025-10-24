
import torch
import torch.nn as nn

ACTIVATIONS = {
    'relu': nn.ReLU,
    'leakyrelu': lambda: nn.LeakyReLU(0.1),
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'silu': nn.SiLU,
    'gelu': nn.GELU,
    'selu': nn.SELU
}

class VGG6(nn.Module):
    """Simplified VGG6 model for CIFAR-10."""
    def __init__(self, num_classes=10, activation='relu', dropout=0.5, batch_norm=False):
        super(VGG6, self).__init__()
        act_fn = ACTIVATIONS.get(activation.lower(), nn.ReLU)

        def act():
            return act_fn() if callable(act_fn) else act_fn

        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act())
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2, 2),
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 256),
            act(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
