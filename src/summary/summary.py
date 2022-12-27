"""
summary.py

Script holds methods for Tensorboard.
"""
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.modules.module import (
    TokenEmbedding,
    PositionEmbedding,
)
from src.config.config import Config


def add_graph(
    model: nn.Module, dataloader: DataLoader, writer: SummaryWriter, config: dict
) -> None:
    """Add graph of model to Tensorboard.

    Args:
        model:
        dataloader:
        writer:
        config:

    """
    device = config.trainer.device
    x_data, _ = next(iter(dataloader))
    writer.add_graph(model=model, input_to_model=x_data.to(device))


def add_position_embedding_weights(
    writer: SummaryWriter, model: nn.Module, global_step: int
) -> None:
    """Adds visualization of position embeddings Tensorboard."""
    for name, module in model.named_modules():
        if isinstance(module, PositionEmbedding):
            embedding = module.embedding.detach().cpu()
            x_min = torch.min(embedding)
            x_max = torch.max(embedding)
            embedding_rescaled = (embedding - x_min) / (x_max - x_min)
            writer.add_image(name, embedding_rescaled, global_step, dataformats="HW")


def add_token_embedding_weights(
    writer: SummaryWriter, model: nn.Module, global_step: int
) -> None:
    """Adds visualization of token embeddings to Tensorboard."""
    for name, module in model.named_modules():
        if isinstance(module, TokenEmbedding):
            embedding = module.embedding.detach().cpu()
            x_min = torch.min(embedding)
            x_max = torch.max(embedding)
            embedding_rescaled = (embedding - x_min) / (x_max - x_min)
            writer.add_image(name, embedding_rescaled, global_step, dataformats="HW")
