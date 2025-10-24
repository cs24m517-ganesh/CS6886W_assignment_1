# VGG6 CIFAR-10
A PyTorch implementation of a lightweight VGG6 network for CIFAR-10 classification.
Includes configurable hyperparameters, W&B logging, and confusion matrix testing.

### Dataset
This project uses the **CIFAR-10** dataset, automatically downloaded via `torchvision.datasets.CIFAR10`.

No dataset files are included in this repository.
If not present, the dataset will be downloaded automatically to `./data/` on first run.

### How to Install depedency
pip install -r requirements.txt

### How to Train Model
python main_train.py \
  --activation gelu \
  --batch_norm true \
  --optimizer adam \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --batch_size 128 \
  --epochs 30 \
  --use_wandb


### How to Evaluate Model
python main_eval.py
python main_eval.py --checkpoint ./checkpoints/best_val_87.19.pth 

### How to Train Model
python main_test.py
python main_test.py --checkpoint ./checkpoints/best_val_87.19.pth 