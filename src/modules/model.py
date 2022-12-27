"""Collection of custom neural networks.

"""
import torch
import torch.nn as nn

from src.config import Config
from .module import MixerBlock, AveragePool


class MlpMixer(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()

        num_blocks = config.module.mlp_mixer.num_blocks
        model_dim = config.module.mlp_mixer.model_dim
        num_classes = config.data.num_classes

        # TODO: Add token embedding.
        # TODO: Add position embedding.

        mixer_blocks = [MixerBlock(config) for _ in range(num_blocks)]
        self.mixer = nn.Sequential(*mixer_blocks)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            AveragePool(dim=-2),
            nn.Linear(in_features=model_dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.mixer(x)
        x = self.mlp_head(x)
        return x