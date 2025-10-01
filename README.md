## ml-common

My own personal common utilities for PyTorch training: losses, datasets, trainer, etc.

## Install

```bash
pip install -e .
```

## Usage

### Trainer

```python
import torch
from ml_common.training import Trainer

# Define your callbacks
def loss_fn(preds, labels):
    return torch.nn.functional.cross_entropy(preds, labels)

def metric_fn(preds, labels):
    acc = (preds.argmax(dim=1) == labels).float().mean()
    return {'accuracy': acc.item()}

# Optional: custom batch preprocessing
def batch_prep_fn(coords, features, labels):
    # Returns: coords, features, batch_ids, labels
    batch_ids = coords[:, 0].long()
    coords = coords[:, 1:4]
    return coords, features, batch_ids, labels

# Setup
cfg = {
    'training_options': {
        'epochs': 100,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 32,
        'precision': 'bf16',  # 'fp16', 'bf16', or 'fp32'
        'grad_clip': 1.0,
        'save_epochs': 10,
        'num_workers': 4,
        'T_max': 100,  # optional, defaults to epochs
    },
    'project_save_dir': './checkpoints'
}

trainer = Trainer(
    model=model,
    device=torch.device('cuda'),
    cfg=cfg,
    loss_fn=loss_fn,
    metric_fn=metric_fn,  # optional
    batch_prep_fn=batch_prep_fn,  # optional
    use_wandb=False
)

# Train
trainer.fit(train_loader, val_loader)

# Test
trainer.test(test_loader)
```

### Losses

```python
from ml_common.losses import (
    angular_distance_loss,
    von_mises_fisher_loss,
    gaussian_nll_loss
)

loss = angular_distance_loss(pred_dirs, true_dirs)
loss = von_mises_fisher_loss(pred_dirs, true_dirs)
loss = gaussian_nll_loss(mu, var, target)
```

### Datasets

```python
from ml_common.data import MmapDataset, create_dataloaders

# Memory-mapped datasets
dataset = MmapDataset(
    mmap_paths='/path/to/data',
    use_summary_stats=True,
    split='train',
    val_split=0.2
)

# Or use the factory
train_loader, val_loader = create_dataloaders(cfg)
```

### Config Example

```python
cfg = {
    'dataloader': 'mmap',  # 'mmap' (auto-detects prometheus/icecube) or 'kaggle'
    'data_options': {
        # Option 1: For memory-mapped dataset formats
        'data_path': '/path/to/data',
        'val_split': 0.2,
        'split_seed': 42,

        # Option 2: Separate train/val paths
        # 'train_data_path': '/path/to/train',
        # 'valid_data_path': '/path/to/val',

        # Common options
        'use_summary_stats': True,
        'batch_size': 256,
        'num_workers': 8,
    },
    'training_options': {
        'epochs': 100,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 256,
        'precision': 'bf16',  # 'fp16', 'bf16', 'fp32'
        'grad_clip': 1.0,
        'save_epochs': 10,
        'num_workers': 4,
        'T_max': 100,  # optional, cosine annealing period
    },
    'project_save_dir': './experiments'
}
```

## Features

- Mixed precision training (fp16/bf16/fp32)
- Automatic checkpointing and logging
- W&B or CSV logging
- Gradient clipping and scaling
- Cosine annealing scheduler
- Memory-mapped datasets for large data
- Custom loss/metric/preprocessing callbacks

## License

MIT
