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

    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()

        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=dims, out_features=hidden_dims),
            nn.GELU(),
            nn.Linear(in_features=hidden_dims, out_features=dims),
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

    def __init__(self, config: Config):
        super().__init__()

        sequence_length = config.model.max_sequence_length
        model_dims = config.model.num_dims  # embedding_dim
        token_dims = config.model.token_dims
        channel_dims = config.model.channel_dims

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(model_dims),
            SwapAxes(axis0=-2, axis1=-1),
            MlpBlock(sequence_length, token_dims),
            SwapAxes(axis0=-2, axis1=-1),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(model_dims),
            MlpBlock(model_dims, channel_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x

class TokenEmbedding(nn.Module):
    """Token embedding module.

    Embeds an integer as a vector of defined dimension.

    Attributes:
        max_sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()

        num_tokens = config.data.num_tokens
        embedding_dim = config.model.num_dims

        size = (num_tokens, embedding_dim)
        embedding = torch.normal(mean=0.0, std=0.01, size=size)
        self.embedding = nn.Parameter(data=embedding, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Receives sequences of token identifiers and returns embedding.

        Args:
            x: Integer tensor holding integer token identifiers.

        Returns:
            Embedded tokens.
        """
        x = self.embedding[x]
        # x = self.embedding(x)  # TODO: use this later with nn.Embedding
        # x = F.embedding(x, self.embedding)  # TODO: Check this.
        return x


class PositionEmbedding(nn.Module):
    """Positional embedding module.

    Positional embedding with different encoding schemes.

    Attributes:
        max_sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()

        embedding_dim = config.model.num_dims
        max_sequence_length = config.model.max_sequence_length

        size = (max_sequence_length, embedding_dim)

        if config.model.position_embedding.encoding == "zeros":
            embedding = torch.zeros(size=size)
        elif config.model.position_embedding.encoding == "ones":
            embedding = torch.ones(size=size)
        elif config.model.position_embedding.encoding == "random_normal":
            embedding = torch.normal(mean=0.0, std=0.01, size=size)
        else:
            raise NotImplementedError(
                f"Embedding {config.model.position_embedding} not implemented."
            )

        requires_grad = True if config.model.position_embedding.is_trainable else False
        self.embedding = nn.Parameter(data=embedding, requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)
        x = x + self.embedding[:sequence_length]
        return x