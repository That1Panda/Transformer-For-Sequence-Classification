import torch
import torch.nn as nn
from transformers import AutoConfig


class Embedding(nn.Module):
    """
    Implements a token and positional embedding layer with layer normalization and dropout.

    Args:
        config (AutoConfig): A configuration object from the Hugging Face transformers library that provides:
            - vocab_size (int): Vocabulary size of the model.
            - hidden_size (int): Dimensionality of the hidden states.
            - max_position_embeddings (int): Maximum sequence length for positional embeddings.
            - layer_norm_eps (float): Epsilon value to avoid division by zero in layer normalization.
            - hidden_dropout_prob (float): Dropout probability.
    """

    def __init__(self, config: AutoConfig):
        super().__init__()

        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional embedding layer
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Layer normalization with epsilon value from config
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer.

        Args:
            input_ids (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output embeddings of shape (batch_size, seq_len, hidden_size).
        """
        # Get the sequence length from input_ids
        seq_length = input_ids.size(1)

        # Create position IDs for the sequence
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

        # Get token embeddings and position embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings

        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings
