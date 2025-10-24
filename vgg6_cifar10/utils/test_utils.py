
import torch
from sklearn.metrics import confusion_matrix
import numpy as np

def test_model(model, loader, device):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, p = outputs.max(1)
            preds.append(p.cpu().numpy())
            labels_all.append(labels.numpy())
    preds = np.concatenate(preds)
    labels_all = np.concatenate(labels_all)
    acc = 100 * (preds == labels_all).mean()
    cm = confusion_matrix(labels_all, preds)
    return acc, cm
