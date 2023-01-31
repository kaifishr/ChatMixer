"""Collection of custom neural networks."""
import torch
import torch.nn as nn

from src.config import Config
from src.modules.module import MixerBlock
from src.modules.module import ConvMixerBlock
from src.modules.module import PositionEmbedding
from src.modules.module import TokenEmbedding
from src.modules.module import Classifier
from src.modules.module import ConvClassifier


class MLPMixer(nn.Module):
    """Character-level isotropic MLP-Mixer."""

    def __init__(self, config: Config):
        """Initializes MLPMixer."""
        super().__init__()

        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)

        num_blocks = config.model.num_blocks
        mixer_blocks = [MixerBlock(config) for _ in range(num_blocks)]
        self.mixer_blocks = nn.Sequential(*mixer_blocks)

        self.classifier = Classifier(config=config)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights for all modules."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.005)
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


class ConvMixer(nn.Module):
    """Character-level isotropic Conv-Mixer."""

    def __init__(self, config: Config):
        """Initializes ConvMixer."""
        super().__init__()

        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)

        sequence_length = config.model.input_sequence_length
        embedding_dim = config.model.embedding_dim
        self.pre_processing = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm([sequence_length, embedding_dim, embedding_dim]),
        )

        num_blocks = config.model.num_blocks
        mixer_blocks = [ConvMixerBlock(config) for _ in range(num_blocks)]
        self.mixer_blocks = nn.Sequential(*mixer_blocks)

        self.classifier = ConvClassifier(config=config)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights for all modules."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.005)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.pre_processing(x)
        x = self.mixer_blocks(x)
        x = self.classifier(x)
        return x