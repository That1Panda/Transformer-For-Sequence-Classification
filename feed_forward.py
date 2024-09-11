import torch
import torch.nn as nn
from transformers import AutoConfig


class FeedForward(nn.Module):
    """
    Implements a feed-forward neural network with two linear layers, GELU activation, and dropout.

    Args:
            - config (transformers.PretrainedConfig): A configuration object from the Hugging Face transformers library.
            - hidden_size (int): The size of the input and output hidden states.
            - intermediate_size (int): The size of the intermediate layer.
            - hidden_dropout_prob (float): Dropout probability for the dropout layer.
    """

    def __init__(self, config: AutoConfig):
        super().__init__()

        # First linear layer mapping hidden_size -> intermediate_size
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)

        # Second linear layer mapping intermediate_size -> hidden_size
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

        # GELU activation function
        self.gelu = nn.GELU()

        # Dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Output tensor after passing through the feed-forward layers, of shape (batch_size, seq_len, hidden_size).
        """
        # Apply the first linear layer
        x = self.linear1(x)

        # Apply GELU activation
        x = self.gelu(x)

        # Apply the second linear layer
        x = self.linear2(x)

        # Apply dropout
        x = self.dropout(x)

        return x
