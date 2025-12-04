import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from typing import Union


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(
    model: nn.Module,
    optimizer_cfg: Union[DictConfig, dict]
) -> torch.optim.Optimizer:
    """
    Create optimizer from config.

    Args:
        model: PyTorch model
        optimizer_cfg: Optimizer configuration

    Returns:
        PyTorch optimizer
    """
    optimizer_name = optimizer_cfg.name.lower() if hasattr(optimizer_cfg, 'name') else optimizer_cfg['name'].lower()
    lr = optimizer_cfg.lr if hasattr(optimizer_cfg, 'lr') else optimizer_cfg['lr']
    weight_decay = optimizer_cfg.weight_decay if hasattr(optimizer_cfg, 'weight_decay') else optimizer_cfg['weight_decay']

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_criterion(criterion_name: str) -> nn.Module:
    """
    Create loss function from config.

    Args:
        criterion_name: Loss function name

    Returns:
        PyTorch loss function
    """
    criterion_name = criterion_name.lower()

    if criterion_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif criterion_name in ['bce', 'bce_with_logits', 'bcewithlogits']:
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


def get_device(device_name: str = 'cuda') -> str:
    """
    Get the appropriate device for training.

    Args:
        device_name: Requested device ('cuda' or 'cpu')

    Returns:
        Available device name
    """
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        return 'cpu'
    return device_name
