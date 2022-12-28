"""Common modules for neural networks."""
import torch
import torch.nn as nn

from src.config import Config


class MlpBlock(nn.Module):

    def __init__(self, dims: int, hidden_dims: int, config: Config) -> None:
        super().__init__()

        dropout_probability = config.model.dropout_probability

        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=dims, out_features=hidden_dims),
            nn.GELU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(in_features=hidden_dims, out_features=dims),
            nn.Dropout(p=dropout_probability)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """MLP Mixer block
    
    Mixes channel and token dimension one after the other.
    """

    def __init__(self, config: Config):
        super().__init__()

        sequence_length = config.model.sequence_length
        embedding_dims = config.model.embedding_dims
        token_dims = config.model.token_dims
        channel_dims = config.model.channel_dims

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dims),
            SwapAxes(axis0=-2, axis1=-1),
            MlpBlock(sequence_length, token_dims, config),
            SwapAxes(axis0=-2, axis1=-1),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dims),
            MlpBlock(embedding_dims, channel_dims, config),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class TokenEmbedding(nn.Module):
    """Token embedding module

    Embeds an integer as a vector of defined dimension.

    Attributes:
        sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()

        num_tokens = config.data.num_tokens
        embedding_dims = config.model.embedding_dims
        dropout_probability = config.model.dropout_probability

        size = (num_tokens, embedding_dims)
        embedding = torch.normal(mean=0.0, std=0.02, size=size)
        self.embedding = nn.Parameter(data=embedding, requires_grad=True)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Receives sequences of token identifiers and returns embedding.

        Args:
            x: Integer tensor holding integer token identifiers.

        Returns:
            Embedded tokens.
        """
        x = self.embedding[x]
        x = self.dropout(x)
        return x


class PositionEmbedding(nn.Module):
    """Positional embedding module.

    Attributes:
        sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()

        sequence_length = config.model.sequence_length
        embedding_dims = config.model.embedding_dims
        dropout_probability = config.model.dropout_probability

        size = (sequence_length, embedding_dims)
        embedding = torch.normal(mean=0.0, std=0.02, size=size)
        self.embedding = nn.Parameter(data=embedding, requires_grad=True)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.embedding
        x = self.dropout(x)
        return x


class AveragePool(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=self.dim)
        return x


class Classifier(nn.Module):

    def __init__(self, config: Config) -> None:
        """Initializes Classifier class."""
        super().__init__()

        sequence_length = config.model.sequence_length
        embedding_dims = config.model.embedding_dims
        # dropout_probability = config.model.dropout_probability
        # hidden_dim = 32

        num_classes = config.data.num_tokens
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dims),
            # Option 0
            # AveragePool(dim=-2),
            # nn.Linear(in_features=hidden_dim, out_features=num_classes)
            # Option 1
            # nn.Linear(in_features=embedding_dims, out_features=hidden_dim)
            # nn.GELU(),
            # nn.Dropout(p=dropout_probability),
            # nn.Flatten(),
            # nn.Linear(in_features=sequence_length * hidden_dim, out_features=num_classes)
            # Option 2
            nn.Flatten(),
            nn.Linear(in_features=sequence_length * embedding_dims, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x