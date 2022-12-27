"""Common modules for neural networks."""
import torch
import torch.nn as nn

from src.config import Config


class AveragePool(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=self.dim)
        return x


class MlpBlock(nn.Module):

    def __init__(self, dim: int, config: Config):
        super().__init__()

        hidden_dim = config.module.mlp_block.hidden_dim

        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.mlp_block(x)
        return x


class SwapAxes(nn.Module):

    def __init__(self, axis0: int, axis1):
        super().__init__()
        self.axis0 = axis0
        self.axis1 = axis1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.swapaxes(x, axis0=self.axis0, axis1=self.axis1)


class MixerBlock(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        model_dim = config.module.mlp_mixer.model_dim

        _, img_height, img_width = config.data.input_shape
        patch_size = config.module.patch_embedding.patch_size
        patch_dim = (img_height // patch_size) * (img_width // patch_size)

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(model_dim),
            SwapAxes(axis0=-2, axis1=-1),
            MlpBlock(patch_dim, config),
            SwapAxes(axis0=-2, axis1=-1),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(model_dim),
            MlpBlock(model_dim, config)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x