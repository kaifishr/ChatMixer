"""Collection of custom neural networks."""
import torch
import torch.nn as nn

from src.config import Config
from src.modules.module import (
    AveragePool, 
    MixerBlock, 
    PositionEmbedding,
    TokenEmbedding, 
)
from src.utils.tools import count_model_parameters


class CharacterMixer(nn.Module):
    """Character-level isotropic MLP-Mixer."""

    def __init__(self, config: Config):
        """Initializes CharacterMixer."""
        super().__init__()

        num_blocks = config.model.num_blocks
        num_dims = config.model.num_dims  # -> embedding_dim
        num_classes = config.data.num_classes
        self.max_sequence_length = config.model.max_sequence_length

        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)

        mixer_blocks = [MixerBlock(config) for _ in range(num_blocks)]
        self.mixer_blocks = nn.Sequential(*mixer_blocks)

        # TODO: Make classifier a class. 
        num_classes = config.data.num_tokens
        use_bias = config.model.classifier.use_bias
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_dims),
            AveragePool(dim=-2),
            nn.Linear(in_features=num_dims, out_features=num_classes, bias=use_bias)
        )

        count_model_parameters(self)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes weights for all modules."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.mixer_blocks(x)
        x = self.classifier(x)
        return x