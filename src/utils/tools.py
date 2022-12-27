"""Script holds tools for network manipulation."""
import os
import random
import numpy as np

from textwrap import wrap

import torch
from torch import nn


def save_checkpoint(model: nn.Module, ckpt_dir: str, model_name: str) -> None:
    """Save model checkpoint.

    Args:
        model:
        ckpt_dir:
        model_name:

    """
    model_path = os.path.join(ckpt_dir, f"{model_name}.pth")
    torch.save(obj=model.state_dict(), f=model_path)


def load_checkpoint(model: nn.Module, ckpt_dir: str, model_name: str) -> None:
    """Load model from checkpoint.

    Args:
        model:
        ckpt_dir:
        model_name:

    """
    model_path = os.path.join(ckpt_dir, f"{model_name}.pth")
    state_dict = torch.load(f=model_path)
    model.load_state_dict(state_dict=state_dict)


def count_model_parameters(
    model: nn.Module, is_trainable: bool = True, verbose: bool = True
) -> int:
    """Counts model parameters.

    Args:
        model: PyTorch model.
        is_trainable: Count only trainable parameters if true.
        verbose: Print number of trainable parameters.

    Returns:
        Number of model parameters.

    """
    n_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad is is_trainable
    )

    if verbose:
        print(
            f"Number of trainable parameters: {'.'.join(wrap(str(n_params)[::-1], 3))[::-1]}."
        )

    return n_params


def set_random_seed(seed: int = 0, is_cuda_deterministic: bool = False) -> None:
    """Controls sources of randomness.

    This method is not bulletproof.
    See also: https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed: Random seed.

    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if is_cuda_deterministic:
        torch.use_deterministic_algorithms(is_cuda_deterministic)
